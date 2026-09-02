# Plan: Tableau Inspector for `cassowary-rs`

Goal: render the solver's internal state as a textbook-style simplex tableau —
basis column on the left, `c_B` beside it, coefficient body in the centre, and
the basic feasible solution (RHS) on the right — so the algorithm can be watched
rather than inferred.

---

## 0. Reality check: what "the tableau" actually is here

Before designing the display, three facts about this implementation constrain
what a faithful rendering can show.

**(a) The tableau is stored in dictionary (solved) form, not as a coefficient matrix.**

`Solver.rows: HashMap<Symbol, Box<Row>>` maps each **basic** symbol to a `Row`
holding `cells: HashMap<Symbol, f64>` plus a `constant`. The invariant is:

```
basic_symbol = constant + Σ_j  cells[j] * nonbasic_j
```

This is the textbook tableau in canonical form w.r.t. the current basis — i.e.
already multiplied through by `B⁻¹` — and it is *stored* that way rather than
recomputed. (Not "revised simplex", despite the term's pull: see §1.1.) There is no
stored `A`, `b`, or `B⁻¹` to display; the rows already are `B⁻¹A` and `B⁻¹b`
(sign-flipped, since the basic symbol has been moved to the left-hand side).

Consequence: the centre body is `-cells[j]` if you want textbook-signed
`B⁻¹A` columns, or `cells[j]` verbatim if you want the dictionary the code
actually manipulates. **Recommendation: show the dictionary signs verbatim**
and label the header accordingly, because that is what `optimise`,
`get_leaving_row` and `substitute` operate on. A sign flip would make the
display prettier and the debugging useless.

**(b) There is no cost vector `c` to index into.**

The objective is itself a `Row` (`Solver.objective`), in the same dictionary
form. Only error symbols ever receive a nonzero cost, at
`create_row` → `objective.insert_symbol(error, constraint.strength())`.
External (user) variables, slack, and dummy symbols all have cost 0.

Consequence: `objective.cells[j]` is **not** `c_j` — it is the *reduced cost*
`z_j − c_j` for nonbasic `j`, because every basic symbol has already been
substituted out. `get_entering_symbol` confirms this: it picks the first symbol
with coefficient `< 0.0`.

**(c) `c_B` is structurally zero in the live objective row.**

Since basic symbols are substituted out of the objective, looking up a basic
symbol in `objective.cells` always yields 0. Printing that column would be
truthful and worthless.

**Adaptation (this is the key design decision):** reconstruct `c_B` from the
*original* cost of each basic symbol. This needs no new solver state — the
mapping already exists in `Solver.cns: HashMap<Constraint, Tag>`. Every error
symbol was created for exactly one constraint, and its cost is that
constraint's `strength()`. So:

```
original_cost(sym) = match sym.type_() {
    Error => strength of the constraint whose Tag has marker == sym or other == sym,
    _     => 0.0
}
```

Build that as a one-shot `HashMap<Symbol, f64>` reverse index at render time.
`O(constraints)`, no hot-path cost, no mutation of `Solver`.

---

## 1. Textbook → cassowary mapping

Notation: `m` constraints, `n` variables, basis matrix `B` (the `m` basic
columns of `A`), `N` the rest.

### 1.1 Matrices and vectors

| Textbook | In this crate | Notes |
|---|---|---|
| `A` (m×n constraint matrix) | **never materialised** | The closest thing is a `Constraint`'s `Expression` (`terms` + `constant`), held in `Solver.cns`. But `create_row` substitutes every already-basic variable *as it reads the terms*, so even the freshly built row is in current nonbasic coordinates. A raw `A` row never exists in storage. |
| `B` (basis matrix) | **never materialised** | Represented only by *which* symbols are keys of `Solver.rows`. |
| `B⁻¹` | **never formed, never applied** | See the note below — this matters more than it looks. |
| `N` (nonbasic columns) | implicit: every symbol appearing in some `row.cells` or in `objective.cells`, minus the basis | |
| `B⁻¹A` (the body) | `-row.cells[j]` | `cells[j] == -a_ij`. The tableau is *stored* in this form; it is never computed from `A`. |
| `b` (original RHS) | not retained | Folded into rows at add time. `suggest_value` mutates `row.constant` in place to simulate changing `b`. |
| `B⁻¹b` (RHS / the BFS) | `row.constant` | this *is* the current value of the basic symbol |
| `x_B` (basic variables) | keys of `Solver.rows` | one row per basic symbol |
| `x_N` (nonbasic variables) | the column set above | all zero at a vertex — that is what makes `rhs` the solution |
| `c` (cost vector) | **does not exist as a vector**; see 1.2 | |
| `c_B` | reconstructed from `cns` + `Tag` (§0c) | nonzero only for basic **Error** symbols |
| `c_N` | `ColumnHeader.cost`, same reconstruction | |
| `y = c_B B⁻¹` (duals / shadow prices) | not stored; derivable as the `Z_j` of a constraint's marker symbol | sign convention needs care — the solver minimises |
| identity submatrix under `B` | **not stored** | implicit in the dictionary form: each row *is* its basic variable, solved for |
| `m` | `Solver.rows.len()` | |
| `n` | `m` + number of nonbasic symbols | |

> **This is not "revised simplex", despite the term's pull.**
> Revised simplex stores `B⁻¹` (or LU factors) alongside `A` and prices columns
> on demand as `B⁻¹A_j`, precisely so it need not update a full tableau each
> pivot. Cassowary does the opposite: `Solver::substitute` loops over **every**
> row plus the objective at each pivot, updating them all in place. That is the
> **standard tableau (dictionary) method**, with a sparse `HashMap` per row
> instead of a dense array. So there is no `B⁻¹` to display because the tableau
> is *permanently kept* in `B⁻¹`-multiplied form, not because the inverse is
> being applied on the fly.

### 1.2 The `C_j` row, and why Error variables replace it

There is no cost vector and no `C_j` row to read. Cost enters the system in
exactly one place — `create_row`, when a constraint's strength is below
`REQUIRED`:

```rust
objective.insert_symbol(error, constraint.strength());
```

So **an Error symbol is the cost mechanism**: it is a variable whose value is
"by how much this constraint is being violated", and whose objective
coefficient is the constraint's strength. Minimising `Σ strength·error` is what
"prefer to satisfy stronger constraints" means operationally.

| | created when | count | cost | in the tableau |
|---|---|---|---|---|
| `Error` | strength `< REQUIRED` | 1 for `<=`/`>=`, **2** for `==` (`errplus`, `errminus`, to allow violation in either direction) | `constraint.strength()` | the only symbols with nonzero `c_j` |
| `Slack` | any `<=` or `>=` | 1 | 0 | coefficient `+1` for `<=`, `-1` for `>=` — one symbol covers both the textbook's *slack* and *surplus* |
| `Dummy` | `==` at `REQUIRED` | 1 | 0 | marker only; must stay nonbasic at zero, and is **exempt from the optimality test** |
| `External` | first use of a user `Variable` | 1 per variable | 0 | the `x_j` you actually care about |

Consequences for the display:

* The `C_j` header row is all zeros except under Error columns.
* `c_B` is nonzero only when an Error variable is *basic* — i.e. only when some
  soft constraint is actually being violated. A basis with no costed rows means
  every soft constraint is currently satisfied.
* Strengths are huge (`REQUIRED = 1_001_001_000`, `STRONG = 1e6`, `MEDIUM = 1e3`,
  `WEAK = 1`), so the `C_j` and `c_B` columns need symbolic formatting (§2).

### 1.3 Tableau display elements

| Textbook | Source | Notes |
|---|---|---|
| Basis label | key of `Solver.rows` | named via `var_for_symbol` + caller's name map |
| `c_B(i)` | reconstruction from `cns` + `Tag` | §0c |
| Body | `-row.cells[j]` | §1b |
| `b` column | `row.constant` | |
| `Z_j` | `Σ_i c_B(i)·a_ij`, computed | |
| `C_j − Z_j` | `objective.cells[j]` **directly** | no sign flip; §1b |
| Objective value `z` | `Σ_i c_B(i)·rhs_i`, computed | **not** `objective.constant` — that drifts across `suggest_value`, §7b.2 |
| Ratio column | `-row.constant / row.cells[entering]` where `cells[entering] < 0` and the row is non-External | mirrors `get_leaving_row` |

### 1.4 Algorithm steps

| Textbook | In this crate | Notes |
|---|---|---|
| Entering column (Dantzig: most positive `C_j − Z_j`) | `get_entering_symbol` — **first** symbol with `cells[j] < 0`, skipping `Dummy` | Bland-like; nondeterministic because it scans a `HashMap` |
| Minimum-ratio test / leaving row | `get_leaving_row` | never picks an External row — user variables are free to move |
| Pivot | `Row::solve_for_symbol` then `Solver::substitute` | `substitute` touches every row *and* the objective |
| Pivot element | `row.cells[entering]` of the leaving row | |
| Optimality test | all non-Dummy `cells[j] >= 0` | §7b.4 |
| Degenerate vertex | a basic row with `rhs == 0` | very common here (`b1.l`, pinned by an `EQ(REQUIRED)| 0`) |
| Unbounded | `get_leaving_row` returns `None` | surfaces as `InternalSolverError` |
| Two-phase / Big-M start | `add_with_artificial_variable`, only when `choose_subject` returns `Invalid` | |
| Phase-I objective `w` | `Solver.artificial`, a second `Row` | `Some` only during that call; feasible iff it optimises to ~0 |
| Dual simplex | `dual_optimise` + `get_dual_entering_symbol` | the `suggest_value` path; holds optimality, restores feasibility |
| Starting basis | **no such step** — see below | |

### 1.5 Things with no textbook counterpart

- **No initial all-slack basis.** The text starts at the origin because that
  basis is trivially feasible. Cassowary never solves from scratch:
  `add_constraint` appends one row to an already-optimal tableau and
  re-optimises, and `choose_subject` prefers an **External** variable as the new
  row's basic symbol, falling back to the slack/error marker.
- **`Tag { marker, other }`** — the two symbols by which a constraint is tracked,
  so it can later be *removed*. Batch simplex never removes a constraint, so
  there is no name for this.
- **Strength** — the whole soft-constraint mechanism. The text has only
  satisfiable-or-infeasible.
- **`Solver.edits` / edit variables** — a constraint whose RHS is meant to be
  changed repeatedly and cheaply.
- **`Solver.infeasible_rows`** — rows queued for `dual_optimise`. Mark them `!`.
- **`should_clear_changes` / `changed`** — incremental change reporting for the
  UI consumer; no algorithmic role.
- **Nonbasic External variables** — value 0, so they get no row. List them
  separately or the printed solution is silently incomplete.

---

## 1b. Rosetta: textbook tableau ↔ cassowary

Reference worked example: [`example.md`](example.md) — `max 7x₁ + 6x₂` subject to
`2x₁ + 4x₂ ≤ 16`, `3x₁ + 2x₂ ≤ 12`, `x ≥ 0`. Optimal at `x₁ = 2, x₂ = 3, z = 32`.

### The conversion rule

Textbook row (canonical form w.r.t. the basis):

```
basic + Σ a_ij · nonbasic_j = b_i
```

Cassowary `Row` (dictionary / solved form):

```
basic = constant + Σ cells[j] · nonbasic_j
```

Rearranging gives the exact, mechanical correspondence:

```
cells[j] == -a_ij            constant == b_i
```

**Negate the body, keep the RHS.** Verified against iteration 3 of `example.md`:
the textbook `x1` row `[1, 0, -1/4, 1/2 | 2]` is
`Row { constant: 2.0, cells: { s1: +1/4, s2: -1/2 } }`.

### Reduced costs and sign convention

**The reduced-cost row is *not* sign-flipped.** (An earlier draft of this plan
claimed it was; that was wrong, and the implementation initially inherited the
error. Corrected here and verified numerically — see below.)

`objective.cells[j]` is `∂z/∂x_j`, which is exactly `c_j − z_j` — the same
formula the text prints as `C_j − Z_j`. Only the **body** needs the
`cells[j] = −a_ij` conversion.

What actually differs between the text and the solver is the **direction of
optimisation**, not the sign of any number:

| | optimal when | entering rule |
|---|---|---|
| `example.md` (a maximisation) | all `C_j − Z_j ≤ 0` | most positive |
| Cassowary (minimises weighted violation) | all `c_j − z_j ≥ 0` | **first** negative |

So a correctly-solved cassowary tableau shows an all-*positive* bottom row and
is optimal — which reads as "not yet optimal" to anyone carrying the max
convention over from the text. The renderer prints a reminder line beneath every
table for this reason.

The apparent sign flip only appears if you *re-encode* a max problem as a min:
that negates every `c_j`, hence every reduced cost. It is an artefact of the
re-encoding, not a difference between the two representations.

Note the pivot rule also differs in kind, not just direction: the text uses
Dantzig's rule (most positive `C_j − Z_j`); `get_entering_symbol` takes the
*first* negative coefficient it meets while iterating a `HashMap`. That is
Bland-like — it avoids cycling — but makes the pivot *sequence* nondeterministic
across processes.

### `Z_j` row and the per-column identity

`Z_j = Σ_i c_B(i) · a_ij` is computable from the reconstructed `c_B` (§0c) and
the sign-converted body, so the inspector can emit the full textbook footer.

That yields the **strongest invariant available**, and it should be the primary
test rather than an afterthought:

```
for every column j:   c_j − Z_j  ==  objective.cells[j]
```

Two independent routes to the same number. It simultaneously validates the `c_B`
reconstruction, the `cells[j] = −a_ij` sign rule, and the substitution
invariant. Its aggregate form is

```
objective.constant  ==  Σ_i c_B(i) · rhs_i
```

which holds because nonbasic variables are zero at a vertex.

### Structural differences that have no textbook counterpart

| Text | Cassowary |
|---|---|
| Fixed m×n grid, every column materialised | Sparse `HashMap<Symbol, f64>`; absent key = 0. A column does not exist until a symbol appears |
| Basis = which columns carry the identity | Basis = the **keys** of `Solver.rows`. The identity submatrix is implicit and never stored |
| `C_j` header row read from a cost vector | No cost vector exists. Only `Error` symbols carry cost, equal to their constraint's `strength()` |
| `Z_j` / `C_j − Z_j` recomputed each iteration | Objective is itself a `Row`, updated incrementally by `substitute` |
| Starts at the all-slack basis (the origin) | **No such phase.** `choose_subject` prefers an External variable; rows are appended one at a time to an already-optimal tableau |
| Slack subscripts index rows; structural subscripts index variables | Same, but both are a single flat `Symbol(usize, SymbolType)` id space allocated from `id_tick` |
| Big-M / two-phase for a hard start | `add_with_artificial_variable`, entered only when `choose_subject` returns `Invalid` |
| Primal simplex only | Plus `dual_optimise` for `suggest_value` — preserves optimality, restores feasibility |

### Render option

`RenderOpts::textbook_signs: bool`.

When set: negate the body (`-cells[j]`), emit a `Z_j` row, label the footer
`Cj-Zj`, and use `+---+` box borders so output can be diffed line-by-line
against `example.md`. The reduced-cost row is **not** negated (see above).

Pair it with `RenderOpts::fractions: bool` — rational approximation with
denominator ≤ 64 and tolerance 1e-9, falling back to decimal. Values like `8/3`
and `3/8` are unreadable as `2.667` / `0.375` when comparing against a text.

---

## 2. Naming

Symbols carry no names — `Symbol(usize, SymbolType)`. Rendering plan:

- `External` → look up `Solver.var_for_symbol[sym] -> Variable`, then consult a
  caller-supplied `HashMap<Variable, String>`. Fall back to `x{id}`.
- `Slack` → `s{id}`, `Error` → `e{id}`, `Dummy` → `d{id}`, `Invalid` → `?`.
- Artificial variables are created as `SymbolType::Slack`
  (`add_with_artificial_variable`), so they are indistinguishable from real
  slack by type alone. Detect them by "is a key of `rows` but appears in no
  constraint `Tag`" and label `a{id}`, or accept the ambiguity and note it.

For the ratatui checkout at `/Users/alex/Repso/ratatui`, `layout.rs` already
holds `vars: HashMap<Variable, (usize, usize)>` mapping each variable to
`(element_index, field)` where field is 0=x, 1=y, 2=width, 3=height. That is a
ready-made name source: `e2.width`, `e0.x`, etc.

**Strength formatting.** Raw strengths are huge (`REQUIRED = 1_001_001_000`,
`STRONG = 1e6`, `MEDIUM = 1e3`, `WEAK = 1`) and will destroy column alignment.
Decompose via the inverse of `strength::create` and print symbolically —
`STRONG`, `2·MEDIUM`, `M+3W` — with raw values available under a verbose flag.

---

## 3. Architecture

Everything needed is **private** to the crate (`Row`, `Symbol`, `Solver` fields),
so this cannot live in an external crate or an `examples/` file. It must be a
module inside `cassowary`.

```
src/
  lib.rs          # add: #[cfg(feature = "tableau")] pub mod tableau;
  solver_impl.rs  # add: snapshot accessor + optional step hook, both gated
  tableau.rs      # NEW — snapshot types, naming, rendering
```

**Feature flag.** Add to `Cargo.toml`:

```toml
[features]
default = []
tableau = []
```

Zero cost when off; the ratatui checkout pinning `cassowary = "0.3"` is
unaffected unless it opts in.

**Note on crate age.** This is 2015-edition code (`try!`, `ATOMIC_USIZE_INIT`,
`extern crate`-style `use {Symbol, ...}` root paths). New code must match that
style or the build breaks. Do **not** run `cargo fix --edition` as part of this
work — modernising and instrumenting at once makes both changes unreviewable.

---

## 4. Data model (`src/tableau.rs`)

Plain owned snapshot types, decoupled from solver internals so a snapshot can
outlive a mutation and snapshots can be diffed:

```rust
pub struct Tableau {
    pub columns: Vec<ColumnHeader>,  // sorted nonbasic symbols
    pub rows: Vec<TableauRow>,       // sorted by basis symbol
    pub objective: ObjectiveRow,
    pub phase_one: Option<ObjectiveRow>,   // Solver.artificial
    pub edits: Vec<(String, f64, f64)>,    // name, suggested value, strength
    pub nonbasic_externals: Vec<String>,   // value 0 by definition
}

pub struct ColumnHeader { pub name: String, pub kind: SymbolKind, pub cost: f64 }
pub struct TableauRow {
    pub basis: String,
    pub kind: SymbolKind,
    pub c_b: f64,
    pub coeffs: Vec<f64>,    // parallel to Tableau::columns, 0.0 for absent
    pub rhs: f64,            // the BFS value
    pub infeasible: bool,
    pub ratio: Option<f64>,  // populated only when an entering column is set
}
pub struct ObjectiveRow { pub reduced: Vec<f64>, pub value: f64 }

pub struct SymbolKind; // mirror of private SymbolType, made public here
```

`SymbolType` is private and `#[derive]`s no `Debug`; mirror it as a public
`SymbolKind` in this module rather than widening the private enum's visibility.

**Column ordering:** sort by `(SymbolKind, id)` — External, Slack, Error, Dummy
— so successive snapshots line up visually and diffs are meaningful. `HashMap`
iteration order is randomised per process; unsorted output would reshuffle
between runs and be useless for step-by-step comparison.

**Accessor** on `Solver`, gated:

```rust
#[cfg(feature = "tableau")]
pub fn tableau(&self, names: &HashMap<Variable, String>) -> Tableau
```

Pure read; borrows `objective` via `RefCell::borrow()` — safe from outside any
solver method, but **not** from inside a step hook fired mid-`optimise`, where
the objective is already mutably borrowed. See §6.

---

## 5. Rendering

`impl fmt::Display for Tableau`, plus `fn render(&self, opts: RenderOpts) -> String`.

Two passes: measure every cell to compute per-column widths, then emit. Fixed
`{:>W.3}` numeric formatting, `-0.000` normalised to `0.000`, and values passing
`near_zero` (|v| < 1e-8) printed as `·` so the sparsity that drives the algorithm
is visible at a glance.

Both modes share the `+---+` box grid. The plan originally sketched a lighter
layout for dictionary mode, but a single renderer that only varies content and
labels is simpler and keeps the two modes diffable against each other, which
turned out to matter more. Target layout:

```
── tableau ── 2 basic / 5 nonbasic ─────────────────────────────────────────
                    c_j │      0        0     1e6      1e6       0
   c_B  BASIS          │   e0.x    e0.wid    e_7      e_8      s_5  │      RHS
  ──────────────────────┼───────────────────────────────────────────┼─────────
     0  e1.x           │  -1.000    0.500      ·   -0.500    1.000  │   50.000
   1e6  e_7          ! │      ·     1.000  1.000       ·        ·   │    0.000
  ──────────────────────┼───────────────────────────────────────────┼─────────
        z_j - c_j       │      ·        ·       ·   -500.0       ·  │  z = 25.000
── nonbasic externals (= 0): e2.width, e2.x
── edits: e0.width = 80.000 (STRONG)
── ! = infeasible row pending dual_optimise
```

`RenderOpts`: `{ width_limit, show_raw_strengths, mark_entering: Option<Symbol>,
elide_zero_columns: bool }`.

**Width is the real risk.** A ratatui 3-pane layout produces on the order of
15–25 variables and 30+ symbols — far past 80 columns. Mitigations, in order of
preference:

1. `elide_zero_columns` — drop columns that are zero in every row *and* in the
   objective. Cheap and usually decisive, since the tableau is sparse.
2. Column paging — emit N columns per block, repeating the `c_B`/BASIS gutter.
3. A `--focus` filter listing symbols of interest.

Do (1) first and measure before building (2) or (3).

---

## 6. Phases

### Phase 1 — static snapshot (the deliverable)

1. Add the `tableau` feature and `src/tableau.rs`.
2. Mirror `SymbolType` as public `SymbolKind`; implement symbol naming.
3. Build the `Symbol -> original cost` reverse index from `cns` + `Tag`.
4. Implement `Solver::tableau(&self, names)` and `Display`.
5. Column elision + width measurement.

Exit criterion: reproduce the two-box example from the `lib.rs` module docs and
print the tableau after each of `add_constraint` × 8, `suggest_value(300)`,
`suggest_value(75)`. The RHS column for external basic symbols must agree with
`Solver::get_value` for every variable, at every step. That is the correctness
check — if RHS and `get_value` ever disagree, the snapshot is wrong.

### Phase 2 — step tracing

Static snapshots between public calls miss the interesting part: `add_constraint`
runs `create_row` → `choose_subject` → `substitute` → `optimise`, and `optimise`
pivots to completion, all before returning. To see individual pivots:

Add a gated hook field to `Solver`:

```rust
#[cfg(feature = "tableau")]
trace: Option<Box<dyn FnMut(TraceEvent, &Tableau)>>,
```

with `TraceEvent::{ RowCreated, SubjectChosen, Pivot { entering, leaving },
ArtificialPhaseStart, ArtificialPhaseEnd, DualPivot, Optimal }`.

**Borrow hazard:** `optimise` holds `objective.borrow_mut()` across the pivot
loop; calling `self.tableau()` from inside would panic on a double borrow, and
`&mut self` is held besides. Resolution: build the snapshot from the already-held
`&Row` rather than re-borrowing — pass the objective row into the snapshot
builder explicitly (`fn snapshot_with_objective(&self, obj: &Row, ...)`) and have
the public `tableau()` be a thin wrapper that borrows and delegates. Decide this
before writing Phase 1's accessor, or Phase 1 will need reworking.

Simpler fallback if the hook proves invasive: a `Vec<Tableau>` recording buffer
that pivot sites push to unconditionally when tracing is enabled, drained by
`take_trace()`. Same borrow constraint applies.

Pivot sites to instrument: `add_constraint` (post-substitute), `optimise` (per
loop iteration), `dual_optimise` (per iteration), `add_with_artificial_variable`
(entry/exit), `remove_constraint` (post `remove_marker_effects`).

## 7c. Phase 2 status: implemented, and what it turned up

**Done.** `Trace` / `TraceStep` / `TraceEvent` / `Phase` in `src/tableau.rs`, the
gated `start_trace` / `stop_trace` / `take_trace` / `is_tracing` accessors and
their internal hooks in `src/solver_impl.rs`, `examples/trace.rs`, and
`tests/trace.rs` (16 tests). All 27 feature tests and both pre-existing tests
pass, with and without the feature; still no new compiler warnings beyond the
crate's 12 pre-existing ones. 60 consecutive full-suite runs, no flakes.

Six things the build settled, several of them corrections to the plan above:

1. **The recording buffer beat the closure hook, for a different reason than
   expected.** The plan's stated borrow hazard — `optimise` holding
   `objective.borrow_mut()` across the loop — *does not exist*. The loop writes
   `Solver::get_entering_symbol(&objective.borrow())`, whose temporary `Ref` is
   dropped at the end of that statement, so no borrow is live at the pivot site.
   And `tableau()` only ever takes *shared* borrows, which `RefCell` grants
   freely, so even a live shared borrow would be harmless.

   The real conflict is `Option<Box<dyn FnMut(TraceEvent, &Tableau)>>` stored in
   `Solver`: invoking it needs `&mut self.trace` while building the snapshot
   needs `&self`. That forces a take-and-replace dance on every call. The
   buffer has no such problem — `trace_snapshot(&self)` and `trace_push(&mut
   self)` simply run in sequence — so the "simpler fallback" is what shipped.
   `snapshot(&Row, ...)` was kept anyway; it costs nothing and `tableau()`
   delegates to it.

2. **Snapshots are taken *before* each pivot, not after.** This was not in the
   plan and matters more than it sounds. `get_leaving_row` **removes** the
   leaving row from `self.rows` before returning it, so a snapshot taken after
   the pivot is chosen is missing a basis row. Taking it before the call also
   makes each step read like a textbook iteration: the tableau, the entering
   column marked `*`, the leaving row marked `<`, the ratio test filled in, and
   the pivot element named. The result of that pivot is the next step's tableau.
   `dual_optimise` needs the same treatment for the same reason.

3. **The artificial-variable heuristic was replaceable, and is now gone.**
   §7b and the original module docs identified artificials heuristically ("a
   slack belonging to no constraint tag"). That is wrong after
   `remove_constraint`, which deletes the tag before snapshotting — every slack
   marker of the removed constraint would render as `a7` rather than `s7`. The
   solver already knows the answer, so a gated `artificial_symbol: Option<Symbol>`
   field now records it for the duration of `add_with_artificial_variable`. The
   label is exact, and `Artificial` correctly appears only in phase-I snapshots.
   This also removed the `tagged` set from the snapshot path entirely
   (`costs_and_tags` → `symbol_costs`).

4. **The dual ratio test runs along a row, not down a column.** It therefore
   cannot use `TableauRow::ratio`. `Tableau::dual_ratios` is parallel to
   `columns` and renders as a row under the reduced costs.

5. **Forcing phase I is harder than it looks** — worth recording, because the
   first three attempts at a test for it passed *vacuously*. `create_row` ends
   with "ensure the row has a positive constant", flipping the sign of any row
   whose constant is negative. That flip usually leaves the marker slack
   negative, which is exactly what `choose_subject` wants, so mixed-direction
   bounds (`a >= 10` then `a <= 20`) and redundant equalities never reach
   `add_with_artificial_variable`. What does reach it is **two bounds in the
   same direction**: after `a >= 10` is basic, the row for `a >= 20` reduces to
   slacks only with the marker positive — no subject, so phase I runs. Note this
   is a pattern a UI layout generates constantly (stacked `Constraint::Min`s),
   so phase I is not an exotic path in ratatui's usage.

6. **The dual trace is the interesting one for UI layout.** `suggest_value` runs
   *no primal iterations at all*: it shifts the right-hand side in place and
   repairs feasibility with dual pivots. `examples/trace.rs` part 2 shows a
   300px resize costing two dual pivots. That is the whole reason cassowary is
   fast enough to re-layout on every frame, and it is invisible without tracing.

Not asserted anywhere, deliberately: the pivot *sequence*. `get_entering_symbol`
takes the first negative-cost symbol out of a `HashMap`, so the route to the
optimum varies per process. Every test in `tests/trace.rs` asserts a *property*
of whatever sequence a run produces — entering column has negative reduced cost,
leaving row attains the minimum ratio, consecutive snapshots are linked by the
pivot the earlier one marks — never a specific sequence.

### Phase 3 — ratatui bridge

An example, `examples/ratatui_layout.rs`, that reconstructs the constraint set
built by `split()` in `/Users/alex/Repso/ratatui/src/layout.rs` for a given
`Layout` + area, names variables from the `(element_index, field)` scheme, and
prints the tableau per pivot. It reconstructs rather than depends: adding a
dev-dependency on a local ratatui checkout would create a circular path
(ratatui depends on cassowary) and is not worth the setup.

This is where the UI-layout insight lands — `Constraint::Min`/`Max`/`Percentage`
each expand into a specific slack/error pattern, and the tableau makes that
expansion legible.

---

## 7. Verification

- **Doc example agreement.** RHS vs `get_value` per variable per step (Phase 1
  exit criterion above).

- **The `example.md` worked example, encoded as a test** (`tests/textbook.rs`).
  Cassowary has no user objective API — it only ever minimises weighted
  constraint violation — so `max 7x₁ + 6x₂` must be *encoded*:

  ```rust
  // structural constraints, inviolable
  x1 |GE(REQUIRED)| 0.0
  x2 |GE(REQUIRED)| 0.0
  2.0*x1 + 4.0*x2 |LE(REQUIRED)| 16.0
  3.0*x1 + 2.0*x2 |LE(REQUIRED)| 12.0
  // objective, encoded as one-sided pull toward an unreachable target.
  // GE below REQUIRED creates slack + error; only the error is costed,
  // so the penalty is 7·max(0, M - x1), i.e. minimising it maximises x1.
  x1 |GE(7.0)| M      // M far outside the feasible region, e.g. 1e6
  x2 |GE(6.0)| M
  ```

  Strengths `7.0` and `6.0` are legal raw values (`WEAK == 1.0`), giving error
  costs of exactly 7 and 6 — matching the text's `C_j`.

  **This will not byte-match `example.md`.** The encoding adds symbols the text
  does not have (dummies for `x ≥ 0`, slack+error pairs for the two objective
  constraints), so the tableau is wider than 2×4. Assert three things instead:

  1. **Solution.** `x1 == 2.0`, `x2 == 3.0` (within `near_zero`).
  2. **Sub-tableau agreement.** Restricted to the `{x1, x2, s1, s2}` columns, the
     basic rows for `x1` and `x2` must match iteration 3 after sign conversion:
     `x2` row `cells == {s1: -3/8, s2: +1/4}, constant == 3`;
     `x1` row `cells == {s1: +1/4, s2: -1/2}, constant == 2`.
  3. **Objective consistency.** `objective.constant == Σ c_B(i)·rhs_i` (§1b).

  If the final basis differs from `{x1, x2}`, that is a genuine finding about the
  encoding, not a bug to force — report it and adjust `M` or the strengths rather
  than weakening the assertion.
- **Existing tests.** `tests/quadrilateral.rs` and `tests/removal.rs` must pass
  unchanged with and without `--features tableau`. This is the guard that
  instrumentation has not perturbed the algorithm.
- **Determinism.** Two runs of the same snapshot must produce byte-identical
  output — the check that column sorting actually defeats `HashMap` ordering.
  Worth a test, since a hash-order leak is invisible in a single run.
- **Dictionary invariant.** For each basic row, assert
  `rhs ≈ constant + Σ coeffs[j] * value_of(nonbasic_j)`, where nonbasic externals
  are 0. Catches sign and substitution errors in the snapshot builder.

## 7b. Phase 1 status: implemented, and what it turned up

**Done.** `src/tableau.rs`, the gated `Solver::tableau` / `Solver::snapshot`
accessors, `examples/textbook.rs`, `examples/two_box.rs`, `tests/textbook.rs`
(9 tests). Both existing tests still pass, with and without the feature; no new
compiler warnings (the 12 that remain are the crate's pre-existing `try!` and
`ATOMIC_USIZE_INIT` deprecations).

`examples/textbook.rs` reproduces iteration 3 of `example.md` exactly: the `x1`
row is `-1/4, 1/2 | 2` and the `x2` row is `3/8, -1/4 | 3` on the `{s5, s6}`
columns, confirming the `cells[j] = -a_ij` rule end to end.

Four corrections to this plan, found by building it:

1. **The reduced-cost row is not sign-flipped** (§1b, rewritten). `objective.cells[j]`
   is already `c_j - z_j`. Only the body converts.

2. **`objective.constant` is not the objective value.** `suggest_value` encodes a
   new right-hand side by shifting row constants directly and never applies the
   matching `c_B^T B^-1 db` correction to the objective row, so the constant
   accumulates drift across edits. Measured: after `suggest_value(300.0)` on a
   STRONG edit variable it reads `300000000` while the true objective is `0`;
   after a further `suggest_value(20.0)` it reads `300000030` against a true `30`.
   Nothing in the solver reads it (`optimise` looks only at `cells`), so the
   drift is harmless to the algorithm - but it must never be displayed as `z`.
   `Tableau::objective_value()` computes `sum(c_B * rhs)` instead; the raw value
   is preserved as `ObjectiveRow::carried_constant`, and the renderer prints a
   note whenever the two disagree.

3. **The aggregate invariant is weaker than the per-column one.**
   `objective.constant == sum(c_B * rhs)` holds only after `add_constraint`.
   The per-column identity `c_j - Z_j == objective.cells[j]` holds *always*,
   including across `suggest_value`, because it involves only coefficients.
   Make that the primary test.

4. **Dummy columns are exempt from the optimality test.** `get_entering_symbol`
   skips `SymbolType::Dummy`: a dummy marks a REQUIRED equality and must stay
   nonbasic at zero. An optimal tableau can therefore show a negative reduced
   cost under a dummy column - the two-box example does, at `d4`. Reading the
   bottom row without this exemption in mind will look like a solver bug.
   `Tableau::is_optimal` and `entering_candidate` skip dummies accordingly.

Also worth knowing before Phase 2: a display bug worth naming because it will
recur. `format!("{}", v as i64)` **truncates toward zero**, so a reduced cost of
`1.9999999999999996` printed as `1`, and an RHS of `2.9999999999999996` as `2`.
Round before casting. Cassowary's arithmetic produces near-integers constantly.

---

## 8. Known limitations to state in the module docs

- The centre body shows dictionary coefficients, not textbook `B⁻¹A`; signs are
  inverted relative to most textbook presentations (§0a).
- `c_B` is reconstructed from constraint strengths, not read from a live cost
  vector — it is the *original* cost, shown for pedagogical continuity (§0c).
- There is no explicit `A`, `B` or `B⁻¹` to display: the tableau is permanently
  stored in `B⁻¹`-multiplied form and updated in place by `substitute` (§1.1).
- Artificial variables are typed as `Slack`; they are identified exactly, from
  the symbol the solver records for the duration of phase I (§7c.3), so the
  `Artificial` label appears only in phase-I snapshots.
- The solver's carried objective constant drifts across `suggest_value`; `z` is
  computed from the basis instead (§7b.2).
- A negative reduced cost under a `Dummy` column does not indicate
  suboptimality (§7b.4).
- Nondeterministic pivot choice (`get_entering_symbol` takes the *first*
  negative-cost symbol from a `HashMap`) means the pivot *sequence* can differ
  between runs even though the final solution is valid. Traces are therefore
  not reproducible across processes, only within one. This is the same
  nondeterminism the crate docs warn about at window width 75.
- A trace snapshots the whole tableau at every pivot, so it is a debugging
  facility, not something to leave enabled.
- `Pivot` and `DualPivot` steps hold the tableau *before* the pivot, marked with
  the element about to be pivoted on (§7c.2); every other event holds the state
  at the moment it is named.
