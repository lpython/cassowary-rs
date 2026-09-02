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
//! * Artificial variables are created with `SymbolType::Slack` and are
//!   identified heuristically (basic, but belonging to no constraint tag).
//! * The solver's carried objective constant is **not** the objective value
//!   after a `suggest_value`: that call encodes a new right-hand side by
//!   shifting row constants directly, with no matching correction to the
//!   objective row. Nothing in the solver reads that constant, so the drift is
//!   harmless there, but this module computes `z` as `sum(c_B * rhs)` instead.

use std::collections::{HashMap, HashSet};
use std::fmt;

use {Row, Symbol, SymbolType, Variable, near_zero};

/// Public mirror of the crate-private `SymbolType`, plus a heuristic
/// `Artificial` case that the private enum does not distinguish.
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

fn kind_of(s: Symbol, tagged: &HashSet<Symbol>) -> SymbolKind {
    let k = SymbolKind::from_type(s.type_());
    // Artificial variables are created as `SymbolType::Slack` in
    // `add_with_artificial_variable`, so type alone cannot distinguish them.
    // A slack belonging to no constraint tag is one. Heuristic.
    if k == SymbolKind::Slack && !tagged.contains(&s) {
        SymbolKind::Artificial
    } else {
        k
    }
}

pub(crate) fn build(
    rows: &HashMap<Symbol, Box<Row>>,
    objective: &Row,
    artificial: Option<&Row>,
    costs: &HashMap<Symbol, f64>,
    tagged: &HashSet<Symbol>,
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
    cols.sort_by_key(|s| (kind_of(*s, tagged).rank(), s.0));

    let columns: Vec<ColumnHeader> = cols
        .iter()
        .map(|s| {
            let kind = kind_of(*s, tagged);
            ColumnHeader {
                name: symbol_name(*s, kind, var_for_symbol, names),
                kind: kind,
                cost: costs.get(s).cloned().unwrap_or(0.0),
            }
        })
        .collect();

    let mut basis: Vec<Symbol> = rows.keys().cloned().collect();
    basis.sort_by_key(|s| (kind_of(*s, tagged).rank(), s.0));

    let infeasible_set: HashSet<Symbol> = infeasible.iter().cloned().collect();

    let out_rows: Vec<TableauRow> = basis
        .iter()
        .map(|s| {
            let row = &rows[s];
            let kind = kind_of(*s, tagged);
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
        hdr.extend(live.iter().map(|&j| self.columns[j].name.clone()));
        hdr.push("b".to_string());
        if want_ratio {
            hdr.push("ratio".to_string());
        }

        lines.push(Line::Sep);
        lines.push(Line::Cells(cj));
        lines.push(Line::Sep);
        lines.push(Line::Cells(hdr));
        lines.push(Line::Sep);

        for r in &self.rows {
            let mut cells = vec![
                format!("{}{}", r.basis, if r.infeasible { " !" } else { "" }),
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

        let mut out = grid(&lines, ncols);

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
