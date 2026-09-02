//! Tableau inspection for the solver, gated behind the `tableau` feature.
//!
//! This renders the solver's internal state as a simplex tableau: basis symbols
//! on the left, `c_B` beside them, the coefficient body in the centre, and the
//! basic feasible solution on the right.
//!
//! # What is and is not faithful to a textbook tableau
//!
//! * The solver stores its tableau in **dictionary (solved) form**: each row is
//!   `basic = constant + sum(cells[j] * nonbasic_j)`. A textbook row is
//!   `basic + sum(a_ij * nonbasic_j) = b_i`. So `cells[j] == -a_ij` and
//!   `constant == b_i`. `RenderOpts::textbook_signs` applies that negation.
//! * There is no cost vector. Only `Error` symbols carry cost, equal to the
//!   `strength()` of the constraint that created them. `c_B` is therefore
//!   *reconstructed* from the solver's constraint map, not read from live state
//!   (basic symbols are substituted out of the objective, so a live lookup would
//!   return zero for every basis row).
//! * `objective.cells[j]` is already `c_j - z_j` - the same formula the text
//!   calls `C_j - Z_j`, with no sign flip. What differs is the optimality
//!   *direction*: the solver minimises, so it is optimal when every reduced
//!   cost is non-negative, and enters on a negative one.
//! * There is no `A`, `B`, `B^-1` or identity submatrix to display. Note this
//!   is *not* revised simplex, which would store `B^-1` and price columns on
//!   demand: `Solver::substitute` updates every row and the objective at each
//!   pivot, so this is the standard tableau method with sparse rows. The
//!   tableau is permanently *kept* in `B^-1`-multiplied form.
//! * `Dummy` columns are exempt from the optimality test. A dummy marks a
//!   REQUIRED equality and must stay nonbasic at zero, so `get_entering_symbol`
//!   skips it. A negative reduced cost under a dummy column does not mean the
//!   tableau is suboptimal.
//! * Artificial variables are created with `SymbolType::Slack`, so the type
//!   alone cannot distinguish one. The solver records the live artificial
//!   symbol while `add_with_artificial_variable` runs, and snapshots use that -
//!   so the `Artificial` label is exact, and appears only in phase-I snapshots
//!   (an artificial is purged from the tableau before that call returns).
//! * `Solver::tableau` only ever sees the state *between* calls. The pivots
//!   themselves happen inside `add_constraint`, `remove_constraint` and
//!   `suggest_value`. `Solver::start_trace` records them - see `Trace`.
//! * The solver's carried objective constant is **not** the objective value
//!   after a `suggest_value`: that call encodes a new right-hand side by
//!   shifting row constants directly, with no matching correction to the
//!   objective row. Nothing in the solver reads that constant, so the drift is
//!   harmless there, but this module computes `z` as `sum(c_B * rhs)` instead.

use std::collections::{HashMap, HashSet};
use std::fmt;

use {Row, Symbol, SymbolType, Variable, near_zero};

/// Public mirror of the crate-private `SymbolType`, plus an `Artificial` case
/// that the private enum does not distinguish (artificials are typed `Slack`).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum SymbolKind {
    External,
    Slack,
    Error,
    Dummy,
    Artificial,
    Invalid,
}

impl SymbolKind {
    fn from_type(t: SymbolType) -> SymbolKind {
        match t {
            SymbolType::External => SymbolKind::External,
            SymbolType::Slack => SymbolKind::Slack,
            SymbolType::Error => SymbolKind::Error,
            SymbolType::Dummy => SymbolKind::Dummy,
            SymbolType::Invalid => SymbolKind::Invalid,
        }
    }
    /// Sort rank, so that columns and rows are grouped External, Slack, Error,
    /// Dummy, Artificial. Ordering must be total and stable: `HashMap`
    /// iteration order is randomised per process, and unsorted output would
    /// reshuffle between runs, making step-by-step comparison useless.
    fn rank(&self) -> u8 {
        match *self {
            SymbolKind::External => 0,
            SymbolKind::Slack => 1,
            SymbolKind::Error => 2,
            SymbolKind::Dummy => 3,
            SymbolKind::Artificial => 4,
            SymbolKind::Invalid => 5,
        }
    }
    fn prefix(&self) -> &'static str {
        match *self {
            SymbolKind::External => "x",
            SymbolKind::Slack => "s",
            SymbolKind::Error => "e",
            SymbolKind::Dummy => "d",
            SymbolKind::Artificial => "a",
            SymbolKind::Invalid => "?",
        }
    }
}

/// One nonbasic column of the tableau.
#[derive(Clone, Debug)]
pub struct ColumnHeader {
    pub name: String,
    pub kind: SymbolKind,
    /// Original objective coefficient `c_j`. Nonzero only for `Error` symbols.
    pub cost: f64,
}

/// One basic row: `basis = rhs + sum(coeffs[j] * column_j)`.
#[derive(Clone, Debug)]
pub struct TableauRow {
    pub basis: String,
    pub kind: SymbolKind,
    /// Original objective coefficient of the basic symbol, reconstructed.
    pub c_b: f64,
    /// Parallel to `Tableau::columns`; absent cells are `0.0`.
    pub coeffs: Vec<f64>,
    /// The current value of the basic symbol: the basic feasible solution.
    pub rhs: f64,
    /// Row is queued in `Solver::infeasible_rows`, pending `dual_optimise`.
    pub infeasible: bool,
    /// Minimum-ratio test result, populated by `Tableau::compute_ratios`.
    pub ratio: Option<f64>,
}

/// An objective row in dictionary form.
#[derive(Clone, Debug)]
pub struct ObjectiveRow {
    /// Reduced costs, parallel to `Tableau::columns`.
    pub reduced: Vec<f64>,
    /// The constant the solver carries in its objective row.
    ///
    /// **This is not reliably the objective value.** `suggest_value` encodes a
    /// new right-hand side by shifting row constants directly, without applying
    /// the matching correction to this constant, so it accumulates drift across
    /// edits. The solver never reads it, so the drift is harmless to the
    /// algorithm - but it is a trap for anyone displaying it as `z`. Use
    /// `Tableau::objective_value` instead.
    pub carried_constant: f64,
}

/// An active edit variable.
#[derive(Clone, Debug)]
pub struct EditView {
    pub name: String,
    pub value: f64,
    pub strength: f64,
}

/// A self-contained snapshot of the solver's tableau.
///
/// Owned and decoupled from solver internals, so a snapshot outlives the
/// mutation that produced it and successive snapshots can be diffed.
#[derive(Clone, Debug)]
pub struct Tableau {
    pub columns: Vec<ColumnHeader>,
    pub rows: Vec<TableauRow>,
    pub objective: ObjectiveRow,
    /// The Phase-I objective, present only mid-`add_with_artificial_variable`.
    pub phase_one: Option<ObjectiveRow>,
    pub edits: Vec<EditView>,
    /// External variables that are not basic, and therefore zero.
    pub nonbasic_externals: Vec<String>,
    /// Entering column of the pivot this snapshot is about to undergo, set on
    /// the pre-pivot snapshots a trace records. `None` outside a trace.
    pub pivot_col: Option<usize>,
    /// Leaving row of that pivot.
    pub pivot_row: Option<usize>,
    /// Dual ratio test results, parallel to `columns`, populated by
    /// `compute_dual_ratios`.
    pub dual_ratios: Option<Vec<Option<f64>>>,
    /// Raw symbol ids behind `columns`, so a pivot can be located by symbol
    /// rather than by display name. Ids are unique across symbol types.
    pub(crate) column_ids: Vec<usize>,
    /// Raw symbol ids behind `rows`.
    pub(crate) row_ids: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

fn symbol_name(
    s: Symbol,
    kind: SymbolKind,
    var_for_symbol: &HashMap<Symbol, Variable>,
    names: &HashMap<Variable, String>,
) -> String {
    if kind == SymbolKind::External {
        if let Some(v) = var_for_symbol.get(&s) {
            if let Some(n) = names.get(v) {
                return n.clone();
            }
        }
    }
    format!("{}{}", kind.prefix(), s.0)
}

/// `artificial` is the artificial variable currently in the basis, if any.
/// Artificial variables are created with `SymbolType::Slack`, so the type alone
/// cannot distinguish one; the solver tracks the symbol itself instead.
fn kind_of(s: Symbol, artificial: Option<Symbol>) -> SymbolKind {
    if artificial == Some(s) {
        SymbolKind::Artificial
    } else {
        SymbolKind::from_type(s.type_())
    }
}

/// Display name for a symbol, matching the names a snapshot uses for the same
/// symbol.
pub(crate) fn label_symbol(
    s: Symbol,
    artificial: Option<Symbol>,
    var_for_symbol: &HashMap<Symbol, Variable>,
    names: &HashMap<Variable, String>,
) -> String {
    symbol_name(s, kind_of(s, artificial), var_for_symbol, names)
}

pub(crate) fn build(
    rows: &HashMap<Symbol, Box<Row>>,
    objective: &Row,
    artificial: Option<&Row>,
    costs: &HashMap<Symbol, f64>,
    artificial_symbol: Option<Symbol>,
    var_for_symbol: &HashMap<Symbol, Variable>,
    names: &HashMap<Variable, String>,
    infeasible: &[Symbol],
    edits: Vec<EditView>,
) -> Tableau {
    // Collect every nonbasic symbol appearing anywhere in the system.
    let mut nonbasic: HashSet<Symbol> = HashSet::new();
    for row in rows.values() {
        for s in row.cells.keys() {
            nonbasic.insert(*s);
        }
    }
    for s in objective.cells.keys() {
        nonbasic.insert(*s);
    }
    if let Some(a) = artificial {
        for s in a.cells.keys() {
            nonbasic.insert(*s);
        }
    }
    for s in rows.keys() {
        nonbasic.remove(s);
    }

    let mut cols: Vec<Symbol> = nonbasic.into_iter().collect();
    cols.sort_by_key(|s| (kind_of(*s, artificial_symbol).rank(), s.0));

    let columns: Vec<ColumnHeader> = cols
        .iter()
        .map(|s| {
            let kind = kind_of(*s, artificial_symbol);
            ColumnHeader {
                name: symbol_name(*s, kind, var_for_symbol, names),
                kind: kind,
                cost: costs.get(s).cloned().unwrap_or(0.0),
            }
        })
        .collect();

    let mut basis: Vec<Symbol> = rows.keys().cloned().collect();
    basis.sort_by_key(|s| (kind_of(*s, artificial_symbol).rank(), s.0));

    let infeasible_set: HashSet<Symbol> = infeasible.iter().cloned().collect();

    let out_rows: Vec<TableauRow> = basis
        .iter()
        .map(|s| {
            let row = &rows[s];
            let kind = kind_of(*s, artificial_symbol);
            TableauRow {
                basis: symbol_name(*s, kind, var_for_symbol, names),
                kind: kind,
                c_b: costs.get(s).cloned().unwrap_or(0.0),
                coeffs: cols.iter().map(|c| row.coefficient_for(*c)).collect(),
                rhs: row.constant,
                infeasible: infeasible_set.contains(s),
                ratio: None,
            }
        })
        .collect();

    let obj = ObjectiveRow {
        reduced: cols.iter().map(|c| objective.coefficient_for(*c)).collect(),
        carried_constant: objective.constant,
    };
    let phase_one = artificial.map(|a| ObjectiveRow {
        reduced: cols.iter().map(|c| a.coefficient_for(*c)).collect(),
        carried_constant: a.constant,
    });

    // External symbols that never became basic are pinned at zero.
    let mut nonbasic_externals: Vec<String> = columns
        .iter()
        .filter(|c| c.kind == SymbolKind::External)
        .map(|c| c.name.clone())
        .collect();
    nonbasic_externals.sort();

    Tableau {
        columns: columns,
        rows: out_rows,
        objective: obj,
        phase_one: phase_one,
        edits: edits,
        nonbasic_externals: nonbasic_externals,
        pivot_col: None,
        pivot_row: None,
        dual_ratios: None,
        column_ids: cols.iter().map(|s| s.0).collect(),
        row_ids: basis.iter().map(|s| s.0).collect(),
    }
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

impl Tableau {
    /// Index of the first column with a negative reduced cost, mirroring
    /// `Solver::get_entering_symbol`.
    ///
    /// Dummy columns are skipped, exactly as the solver skips them: a dummy is
    /// the marker of a REQUIRED equality and must stay nonbasic at zero, so its
    /// reduced cost says nothing about optimality. A tableau showing a negative
    /// reduced cost under a dummy column is still optimal.
    ///
    /// This will not generally agree with the solver's own choice: the solver
    /// scans a `HashMap`, whose order is randomised per process, whereas these
    /// columns are sorted. It identifies *a* valid entering candidate, not
    /// necessarily the one the solver would take.
    pub fn entering_candidate(&self) -> Option<usize> {
        self.columns.iter().enumerate().position(|(j, c)| {
            c.kind != SymbolKind::Dummy
                && c.kind != SymbolKind::Invalid
                && self.objective.reduced[j] < 0.0
                && !near_zero(self.objective.reduced[j])
        })
    }

    /// Whether the tableau is optimal: no non-dummy column has a negative
    /// reduced cost.
    pub fn is_optimal(&self) -> bool {
        self.entering_candidate().is_none()
    }

    /// Fill in the minimum-ratio column for a given entering column, mirroring
    /// `Solver::get_leaving_row`: only rows whose basis is not External and
    /// whose coefficient in that column is negative can leave.
    pub fn compute_ratios(&mut self, entering: usize) {
        for row in &mut self.rows {
            let coeff = row.coeffs[entering];
            row.ratio = if row.kind != SymbolKind::External && coeff < 0.0 {
                Some(-row.rhs / coeff)
            } else {
                None
            };
        }
    }

    /// Fill in the dual ratio row for a given leaving row, mirroring
    /// `Solver::get_dual_entering_symbol`: a column can enter only if it is
    /// non-dummy and its coefficient in the leaving row is *positive*, and the
    /// entering column is the one minimising `reduced_cost / coefficient`.
    ///
    /// Note this ratio runs along a row, not down a column, which is why it is
    /// stored parallel to `columns` rather than in `TableauRow::ratio`.
    pub fn compute_dual_ratios(&mut self, leaving_row: usize) {
        let ratios: Vec<Option<f64>> = {
            let row = &self.rows[leaving_row];
            (0..self.columns.len())
                .map(|j| {
                    if self.columns[j].kind != SymbolKind::Dummy && row.coeffs[j] > 0.0 {
                        Some(self.objective.reduced[j] / row.coeffs[j])
                    } else {
                        None
                    }
                })
                .collect()
        };
        self.dual_ratios = Some(ratios);
    }

    /// Locate a symbol id among the columns.
    pub(crate) fn column_of_id(&self, id: usize) -> Option<usize> {
        self.column_ids.iter().position(|&i| i == id)
    }

    /// Locate a symbol id among the basis rows.
    pub(crate) fn row_of_id(&self, id: usize) -> Option<usize> {
        self.row_ids.iter().position(|&i| i == id)
    }

    /// Mark the primal pivot this snapshot is about to undergo, and run the
    /// minimum-ratio test on the entering column.
    pub(crate) fn mark_pivot(&mut self, entering_id: usize, leaving_id: usize) {
        self.pivot_col = self.column_of_id(entering_id);
        self.pivot_row = self.row_of_id(leaving_id);
        if let Some(c) = self.pivot_col {
            self.compute_ratios(c);
        }
    }

    /// Mark the dual pivot this snapshot is about to undergo, and run the dual
    /// ratio test along the leaving row.
    pub(crate) fn mark_dual_pivot(&mut self, entering_id: usize, leaving_id: usize) {
        self.pivot_col = self.column_of_id(entering_id);
        self.pivot_row = self.row_of_id(leaving_id);
        if let Some(r) = self.pivot_row {
            self.compute_dual_ratios(r);
        }
    }

    /// The coefficient the pivot will divide through by, in dictionary signs.
    /// `None` unless both `pivot_row` and `pivot_col` are set.
    pub fn pivot_element(&self) -> Option<f64> {
        match (self.pivot_row, self.pivot_col) {
            (Some(r), Some(c)) => Some(self.rows[r].coeffs[c]),
            _ => None,
        }
    }

    /// The objective value, `sum(c_B(i) * rhs_i)`.
    ///
    /// Computed from the basis rather than read from the solver's carried
    /// constant, which drifts across `suggest_value` - see
    /// `ObjectiveRow::carried_constant`. Correct because nonbasic variables are
    /// zero at a vertex.
    pub fn objective_value(&self) -> f64 {
        self.rows.iter().map(|r| r.c_b * r.rhs).sum()
    }

    /// Deprecated spelling of `objective_value`.
    pub fn objective_from_basis(&self) -> f64 {
        self.objective_value()
    }

    /// How far the solver's carried objective constant has drifted from the
    /// true objective value. Nonzero after any `suggest_value`.
    pub fn carried_constant_drift(&self) -> f64 {
        self.objective.carried_constant - self.objective_value()
    }

    /// `Z_j = sum(c_B(i) * a_ij)`, in textbook signs.
    pub fn zj(&self) -> Vec<f64> {
        (0..self.columns.len())
            .map(|j| self.rows.iter().map(|r| r.c_b * -r.coeffs[j]).sum())
            .collect()
    }

    /// Columns that are zero in every row and in the objective.
    fn live_columns(&self) -> Vec<usize> {
        (0..self.columns.len())
            .filter(|&j| {
                !near_zero(self.objective.reduced[j])
                    || self.rows.iter().any(|r| !near_zero(r.coeffs[j]))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Rendering options. `RenderOpts::dictionary()` shows the solver's own
/// representation; `RenderOpts::textbook()` converts to the presentation used in
/// most linear-programming texts, for line-by-line comparison.
#[derive(Clone, Debug)]
pub struct RenderOpts {
    /// Negate the body (`cells[j] -> a_ij`) and label the footer `Cj-Zj`.
    /// The reduced-cost row is *not* negated - see the module docs.
    pub textbook_signs: bool,
    /// Emit a `Zj` row above the reduced costs. Requires `textbook_signs`.
    pub show_zj: bool,
    /// Print small rationals as fractions (`8/3`) instead of decimals.
    pub fractions: bool,
    /// Drop columns that are zero everywhere.
    pub elide_zero_columns: bool,
    /// Print strengths as raw numbers rather than `REQ` / `S` / `M` / `W`.
    pub show_raw_strengths: bool,
    /// Include the minimum-ratio column, if `compute_ratios` has been called.
    pub show_ratios: bool,
}

impl RenderOpts {
    pub fn dictionary() -> RenderOpts {
        RenderOpts {
            textbook_signs: false,
            show_zj: false,
            fractions: false,
            elide_zero_columns: true,
            show_raw_strengths: false,
            show_ratios: true,
        }
    }
    pub fn textbook() -> RenderOpts {
        RenderOpts {
            textbook_signs: true,
            show_zj: true,
            fractions: true,
            elide_zero_columns: true,
            show_raw_strengths: true,
            show_ratios: true,
        }
    }
}

impl Default for RenderOpts {
    fn default() -> RenderOpts {
        RenderOpts::dictionary()
    }
}

/// Approximate `v` as `p/q` with `q <= 64`, returning `None` if no such
/// rational is within `1e-9`.
fn as_fraction(v: f64) -> Option<String> {
    if !v.is_finite() {
        return None;
    }
    for q in 1..65u64 {
        let p = (v * q as f64).round();
        if (v - p / q as f64).abs() < 1e-9 {
            return Some(if q == 1 {
                format!("{}", p as i64)
            } else {
                format!("{}/{}", p as i64, q)
            });
        }
    }
    None
}

fn fmt_num(v: f64, opts: &RenderOpts) -> String {
    if near_zero(v) {
        return "0".to_string();
    }
    if opts.fractions {
        if let Some(f) = as_fraction(v) {
            return f;
        }
    }
    if !v.is_finite() {
        return format!("{}", v);
    }
    if (v - v.round()).abs() < 1e-9 && v.abs() < 1e15 {
        // `v as i64` truncates toward zero: 1.9999999999999996 would print as
        // "1". Round first.
        return format!("{}", v.round() as i64);
    }
    format!("{:.3}", v)
}

/// Decompose a strength back into its STRONG/MEDIUM/WEAK bands.
fn fmt_strength(s: f64, opts: &RenderOpts) -> String {
    if near_zero(s) {
        return "0".to_string();
    }
    if opts.show_raw_strengths {
        return fmt_num(s, opts);
    }
    if s >= ::strength::REQUIRED {
        return "REQ".to_string();
    }
    let a = (s / 1_000_000.0).floor();
    let b = ((s - a * 1_000_000.0) / 1000.0).floor();
    let c = s - a * 1_000_000.0 - b * 1000.0;
    let mut parts: Vec<String> = Vec::new();
    for &(v, letter) in &[(a, "S"), (b, "M"), (c, "W")] {
        if v > 0.0 {
            parts.push(if (v - 1.0).abs() < 1e-9 {
                letter.to_string()
            } else {
                format!("{}{}", fmt_num(v, opts), letter)
            });
        }
    }
    if parts.is_empty() {
        "0".to_string()
    } else {
        parts.join("+")
    }
}

enum Line {
    Sep,
    Cells(Vec<String>),
}

fn grid(lines: &[Line], ncols: usize) -> String {
    let mut width = vec![0usize; ncols];
    for l in lines {
        if let Line::Cells(ref cs) = *l {
            for (i, c) in cs.iter().enumerate() {
                if c.chars().count() > width[i] {
                    width[i] = c.chars().count();
                }
            }
        }
    }
    let sep: String = {
        let mut s = String::from("+");
        for w in &width {
            s.push_str(&"-".repeat(w + 2));
            s.push('+');
        }
        s
    };
    let mut out = String::new();
    for l in lines {
        match *l {
            Line::Sep => {
                out.push_str(&sep);
                out.push('\n');
            }
            Line::Cells(ref cs) => {
                out.push('|');
                for i in 0..ncols {
                    let c = cs.get(i).map(|s| s.as_str()).unwrap_or("");
                    let pad = width[i] - c.chars().count();
                    // Column 0 (basis labels) centres; everything else right-aligns.
                    if i == 0 {
                        let l = pad / 2;
                        out.push_str(&format!(" {}{}{} |", " ".repeat(l), c, " ".repeat(pad - l)));
                    } else {
                        out.push_str(&format!(" {}{} |", " ".repeat(pad), c));
                    }
                }
                out.push('\n');
            }
        }
    }
    out
}

impl Tableau {
    pub fn render(&self, opts: &RenderOpts) -> String {
        let live: Vec<usize> = if opts.elide_zero_columns {
            self.live_columns()
        } else {
            (0..self.columns.len()).collect()
        };
        let sign = if opts.textbook_signs { -1.0 } else { 1.0 };
        let want_ratio = opts.show_ratios && self.rows.iter().any(|r| r.ratio.is_some());

        // Column layout: Basis | CB | <variables> | b | [ratio]
        let ncols = 2 + live.len() + 1 + if want_ratio { 1 } else { 0 };
        let mut lines: Vec<Line> = Vec::new();

        let mut cj: Vec<String> = vec![String::new(), "Cj".to_string()];
        cj.extend(live.iter().map(|&j| fmt_strength(self.columns[j].cost, opts)));
        cj.push(String::new());
        if want_ratio {
            cj.push(String::new());
        }

        let mut hdr: Vec<String> = vec!["Basis".to_string(), "CB".to_string()];
        // A trailing `*` marks the entering column of the pivot this snapshot
        // is about to undergo.
        hdr.extend(live.iter().map(|&j| {
            if self.pivot_col == Some(j) {
                format!("{}*", self.columns[j].name)
            } else {
                self.columns[j].name.clone()
            }
        }));
        hdr.push("b".to_string());
        if want_ratio {
            hdr.push("ratio".to_string());
        }

        lines.push(Line::Sep);
        lines.push(Line::Cells(cj));
        lines.push(Line::Sep);
        lines.push(Line::Cells(hdr));
        lines.push(Line::Sep);

        for (i, r) in self.rows.iter().enumerate() {
            let mut cells = vec![
                format!(
                    "{}{}{}",
                    r.basis,
                    if r.infeasible { " !" } else { "" },
                    // A trailing `<` marks the leaving row of the pending pivot.
                    if self.pivot_row == Some(i) { " <" } else { "" }
                ),
                fmt_strength(r.c_b, opts),
            ];
            cells.extend(live.iter().map(|&j| fmt_num(sign * r.coeffs[j], opts)));
            cells.push(fmt_num(r.rhs, opts));
            if want_ratio {
                cells.push(match r.ratio {
                    Some(v) => fmt_num(v, opts),
                    None => "-".to_string(),
                });
            }
            lines.push(Line::Cells(cells));
        }
        lines.push(Line::Sep);

        if opts.show_zj && opts.textbook_signs {
            let zj = self.zj();
            let mut cells = vec!["Zj".to_string(), String::new()];
            cells.extend(live.iter().map(|&j| fmt_num(zj[j], opts)));
            cells.push(fmt_num(self.objective_value(), opts));
            if want_ratio {
                cells.push(String::new());
            }
            lines.push(Line::Cells(cells));
            lines.push(Line::Sep);
        }

        let label = if opts.textbook_signs { "Cj-Zj" } else { "cj-zj" };
        let mut cells = vec![label.to_string(), String::new()];
        // Not negated, unlike the body: `objective.reduced[j]` is already
        // `c_j - z_j`, the same quantity the text calls `Cj - Zj`. Only the
        // *body* needs the dictionary-to-textbook sign conversion.
        cells.extend(
            live.iter()
                .map(|&j| fmt_num(self.objective.reduced[j], opts)),
        );
        cells.push(if opts.show_zj && opts.textbook_signs {
            String::new()
        } else {
            format!("z={}", fmt_num(self.objective_value(), opts))
        });
        if want_ratio {
            cells.push(String::new());
        }
        lines.push(Line::Cells(cells));
        lines.push(Line::Sep);

        // The dual ratio test runs along the leaving row, so it prints as a
        // row beneath the reduced costs rather than as the `ratio` column.
        if let Some(ref dr) = self.dual_ratios {
            let mut cells = vec!["dual ratio".to_string(), String::new()];
            cells.extend(live.iter().map(|&j| match dr[j] {
                Some(v) => fmt_num(v, opts),
                None => "-".to_string(),
            }));
            cells.push(String::new());
            if want_ratio {
                cells.push(String::new());
            }
            lines.push(Line::Cells(cells));
            lines.push(Line::Sep);
        }

        let mut out = grid(&lines, ncols);

        if self.pivot_col.is_some() || self.pivot_row.is_some() {
            let entering = self
                .pivot_col
                .map(|j| self.columns[j].name.as_str())
                .unwrap_or("?");
            let leaving = self
                .pivot_row
                .map(|i| self.rows[i].basis.as_str())
                .unwrap_or("?");
            out.push_str(&format!(
                "  pivot: {}* enters, {} < leaves",
                entering, leaving
            ));
            match self.pivot_element() {
                Some(p) => out.push_str(&format!(
                    ", pivot element = {}\n",
                    fmt_num(sign * p, opts)
                )),
                None => out.push('\n'),
            }
            out.push_str("  (this tableau is the state *before* that pivot)\n");
        }

        if let Some(ref p1) = self.phase_one {
            out.push_str(&format!(
                "  phase-I objective: w={} (artificial variable active)\n",
                fmt_num(p1.carried_constant, opts)
            ));
        }
        if !self.nonbasic_externals.is_empty() {
            out.push_str(&format!(
                "  nonbasic externals (=0): {}\n",
                self.nonbasic_externals.join(", ")
            ));
        }
        for e in &self.edits {
            out.push_str(&format!(
                "  edit: {} = {} ({})\n",
                e.name,
                fmt_num(e.value, opts),
                fmt_strength(e.strength, &RenderOpts::dictionary())
            ));
        }
        out.push_str(&format!(
            "  minimisation: optimal when all cj-zj >= 0, dummy columns exempt \
             (currently {})\n",
            if self.is_optimal() { "optimal" } else { "not optimal" }
        ));
        if !near_zero(self.carried_constant_drift()) {
            out.push_str(&format!(
                "  note: z is computed as sum(c_B*rhs); the solver's carried \
                 objective constant reads {} (drifts across suggest_value)\n",
                fmt_num(self.objective.carried_constant, opts)
            ));
        }
        if self.rows.iter().any(|r| r.infeasible) {
            out.push_str("  ! = infeasible row, pending dual_optimise\n");
        }
        if opts.elide_zero_columns && live.len() < self.columns.len() {
            out.push_str(&format!(
                "  {} all-zero column(s) elided\n",
                self.columns.len() - live.len()
            ));
        }
        out
    }
}

impl fmt::Display for Tableau {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.render(&RenderOpts::default()))
    }
}

// ---------------------------------------------------------------------------
// Step tracing
// ---------------------------------------------------------------------------

/// Which simplex phase a pivot belongs to.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Driving an artificial variable out of the basis, inside
    /// `add_with_artificial_variable`. The objective being minimised is the
    /// artificial row, not the real one.
    One,
    /// Minimising the real objective.
    Two,
}

impl Phase {
    pub fn label(&self) -> &'static str {
        match *self {
            Phase::One => "phase I",
            Phase::Two => "phase II",
        }
    }
}

/// What the solver was doing when a `TraceStep` was captured.
///
/// Symbols are carried as display names rather than `Symbol`s, which are
/// crate-private. Names match the ones in the accompanying `Tableau`.
#[derive(Clone, Debug)]
pub enum TraceEvent {
    /// `add_constraint` created a row and solved it for `subject`, which is now
    /// basic. Captured before the optimisation that follows.
    SubjectChosen { subject: String },
    /// No subject could be chosen, so `add_with_artificial_variable` put
    /// `artificial` into the basis and started phase I.
    ArtificialPhaseStart { artificial: String },
    /// Phase I finished. `success` is whether the artificial objective reached
    /// zero, i.e. whether the constraint was satisfiable.
    ArtificialPhaseEnd { success: bool },
    /// A primal pivot chosen by `optimise`. The snapshot is *pre-pivot*.
    Pivot {
        entering: String,
        leaving: String,
        phase: Phase,
    },
    /// `optimise` found no column with a negative reduced cost and returned.
    Optimal { phase: Phase },
    /// `suggest_value` shifted the right-hand side. Rows driven negative are
    /// marked infeasible and the dual pivots below follow.
    ValueSuggested { variable: String, value: f64 },
    /// A dual pivot chosen by `dual_optimise`. The snapshot is *pre-pivot*.
    DualPivot { entering: String, leaving: String },
    /// `dual_optimise` drained its infeasible queue.
    Feasible,
    /// `remove_constraint` pivoted a marker out of the basis and dropped its
    /// row. Captured before the re-optimisation that follows.
    MarkerRemoved { marker: String },
}

impl TraceEvent {
    /// A one-line description, used as the step header when rendering.
    pub fn describe(&self) -> String {
        match *self {
            TraceEvent::SubjectChosen { ref subject } => {
                format!("row added, solved for {}", subject)
            }
            TraceEvent::ArtificialPhaseStart { ref artificial } => format!(
                "phase I begins: artificial variable {} entered the basis",
                artificial
            ),
            TraceEvent::ArtificialPhaseEnd { success } => format!(
                "phase I ends: artificial objective {}",
                if success {
                    "reached zero (constraint satisfiable)"
                } else {
                    "did not reach zero (constraint unsatisfiable)"
                }
            ),
            TraceEvent::Pivot {
                ref entering,
                ref leaving,
                phase,
            } => format!(
                "{} pivot: {} enters, {} leaves",
                phase.label(),
                entering,
                leaving
            ),
            TraceEvent::Optimal { phase } => format!(
                "{} optimal: no non-dummy column has a negative reduced cost",
                phase.label()
            ),
            TraceEvent::ValueSuggested {
                ref variable,
                value,
            } => format!(
                "suggest_value({}, {}): right-hand side shifted",
                variable, value
            ),
            TraceEvent::DualPivot {
                ref entering,
                ref leaving,
            } => format!(
                "dual pivot: {} leaves (infeasible), {} enters",
                leaving, entering
            ),
            TraceEvent::Feasible => "dual simplex done: no infeasible rows remain".to_string(),
            TraceEvent::MarkerRemoved { ref marker } => format!(
                "constraint removed: marker {} pivoted out and its row dropped",
                marker
            ),
        }
    }

    /// Whether this event is a pivot, primal or dual.
    pub fn is_pivot(&self) -> bool {
        match *self {
            TraceEvent::Pivot { .. } | TraceEvent::DualPivot { .. } => true,
            _ => false,
        }
    }
}

/// One recorded moment: what happened, and the tableau at that moment.
#[derive(Clone, Debug)]
pub struct TraceStep {
    pub event: TraceEvent,
    /// The tableau when the event was captured.
    ///
    /// For `Pivot` and `DualPivot` this is the tableau **before** the pivot is
    /// applied, with `Tableau::pivot_col` and `pivot_row` marking the chosen
    /// element and the relevant ratio test filled in. So a pivot step reads as
    /// "here is the tableau, and here is the pivot about to happen"; the result
    /// of that pivot is the tableau of the next step. Every other event
    /// captures the state at the moment it is named.
    pub tableau: Tableau,
}

impl TraceStep {
    pub fn render(&self, opts: &RenderOpts) -> String {
        format!("{}\n{}", self.event.describe(), self.tableau.render(opts))
    }
}

/// A recording of the solver's pivots.
///
/// Start one with `Solver::start_trace`, and take it back with
/// `Solver::take_trace` or `Solver::stop_trace`.
///
/// **The pivot sequence is not reproducible across processes.**
/// `Solver::get_entering_symbol` takes the first negative-cost symbol out of a
/// `HashMap`, whose iteration order is randomised per process, so two runs of
/// the same program can reach the same optimum by different routes. Within one
/// process a trace is exact.
#[derive(Clone, Debug)]
pub struct Trace {
    pub steps: Vec<TraceStep>,
    pub(crate) names: HashMap<Variable, String>,
}

impl Trace {
    pub(crate) fn new(names: HashMap<Variable, String>) -> Trace {
        Trace {
            steps: Vec::new(),
            names: names,
        }
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Just the pivot steps, primal and dual.
    pub fn pivots(&self) -> Vec<&TraceStep> {
        self.steps.iter().filter(|s| s.event.is_pivot()).collect()
    }

    /// Render every step in order, numbered.
    pub fn render(&self, opts: &RenderOpts) -> String {
        let mut out = String::new();
        let n = self.steps.len();
        for (i, step) in self.steps.iter().enumerate() {
            out.push_str(&format!(
                "--- step {} of {}: {} {}\n",
                i + 1,
                n,
                step.event.describe(),
                "-".repeat(8)
            ));
            out.push_str(&step.tableau.render(opts));
            out.push('\n');
        }
        out
    }
}

impl fmt::Display for Trace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.render(&RenderOpts::default()))
    }
}
