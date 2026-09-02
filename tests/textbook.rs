//! The worked example from `example.md`, encoded and checked against the
//! tableau inspector.
//!
//!   max 7*x1 + 6*x2   s.t.  2*x1 + 4*x2 <= 16,  3*x1 + 2*x2 <= 12,  x >= 0
//!   optimum: x1 = 2, x2 = 3
#![cfg(feature = "tableau")]

extern crate cassowary;

use std::collections::HashMap;

use cassowary::{Solver, Variable};
use cassowary::WeightedRelation::*;
use cassowary::strength::REQUIRED;
use cassowary::tableau::{RenderOpts, Tableau};

const M: f64 = 1000.0;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn setup() -> (Solver, Variable, Variable, HashMap<Variable, String>) {
    let x1 = Variable::new();
    let x2 = Variable::new();
    let mut names = HashMap::new();
    names.insert(x1, "x1".to_string());
    names.insert(x2, "x2".to_string());

    let mut solver = Solver::new();
    // Constraint order fixes symbol ids (`id_tick` is a plain counter), so
    // s5 is the slack of `2x1+4x2<=16` (the text's s1) and s6 that of
    // `3x1+2x2<=12` (the text's s2).
    solver.add_constraints(&[
        x1 |GE(REQUIRED)| 0.0,
        x2 |GE(REQUIRED)| 0.0,
        2.0 * x1 + 4.0 * x2 |LE(REQUIRED)| 16.0,
        3.0 * x1 + 2.0 * x2 |LE(REQUIRED)| 12.0,
    ]).unwrap();
    // Objective, encoded as costed error variables: minimising 7*(M - x1)
    // maximises 7*x1.
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    (solver, x1, x2, names)
}

fn col(t: &Tableau, name: &str) -> usize {
    t.columns.iter().position(|c| c.name == name)
        .unwrap_or_else(|| panic!("no column {}; have {:?}",
                                 name,
                                 t.columns.iter().map(|c| &c.name).collect::<Vec<_>>()))
}

fn row<'a>(t: &'a Tableau, name: &str) -> &'a cassowary::tableau::TableauRow {
    t.rows.iter().find(|r| r.basis == name)
        .unwrap_or_else(|| panic!("{} is not basic; basis is {:?}",
                                  name,
                                  t.rows.iter().map(|r| &r.basis).collect::<Vec<_>>()))
}

#[test]
fn solution_matches_text() {
    let (solver, x1, x2, _) = setup();
    assert!(close(solver.get_value(x1), 2.0), "x1 = {}", solver.get_value(x1));
    assert!(close(solver.get_value(x2), 3.0), "x2 = {}", solver.get_value(x2));
}

/// The strongest available invariant: for every column, `c_j - Z_j` computed
/// from the reconstructed `c_B` and the sign-converted body must equal the
/// reduced cost the solver maintains in its objective row. Two independent
/// routes to the same number, so this validates the `c_B` reconstruction, the
/// `cells[j] == -a_ij` sign rule, and the substitution invariant at once.
#[test]
fn reduced_costs_agree_with_basis() {
    let (solver, _, _, names) = setup();
    let t = solver.tableau(&names);
    let zj = t.zj();
    for (j, c) in t.columns.iter().enumerate() {
        let from_basis = c.cost - zj[j];
        assert!(close(from_basis, t.objective.reduced[j]),
                "column {}: c_j - Z_j = {} but reduced cost = {}",
                c.name, from_basis, t.objective.reduced[j]);
    }
}

/// At a vertex every nonbasic variable is zero, so the objective value is
/// exactly `sum(c_B(i) * rhs_i)`. The solver's own carried constant agrees with
/// it here because only `add_constraint` has run - see `carried_constant_drifts`
/// for where it stops agreeing.
#[test]
fn objective_value_matches_basis() {
    let (solver, _, _, names) = setup();
    let t = solver.tableau(&names);
    assert!(close(t.objective.carried_constant, t.objective_value()),
            "carried constant = {} but sum(c_B * rhs) = {}",
            t.objective.carried_constant, t.objective_value());
    // 7*(1000-2) + 6*(1000-3)
    assert!(close(t.objective_value(), 12968.0), "z = {}", t.objective_value());
}

/// Restricted to the `{s1, s2}` columns, the final basic rows for x1 and x2
/// must reproduce iteration 3 of `example.md` after the `cells[j] == -a_ij`
/// sign conversion.
#[test]
fn sub_tableau_matches_iteration_3() {
    let (solver, _, _, names) = setup();
    let t = solver.tableau(&names);
    let (s1, s2) = (col(&t, "s5"), col(&t, "s6"));

    // text: | x1 | 7 | 1 | 0 | -1/4 | 1/2 | 2 |
    let r = row(&t, "x1");
    assert!(close(r.rhs, 2.0), "x1 rhs = {}", r.rhs);
    assert!(close(-r.coeffs[s1], -0.25), "x1/s1 a_ij = {}", -r.coeffs[s1]);
    assert!(close(-r.coeffs[s2], 0.5), "x1/s2 a_ij = {}", -r.coeffs[s2]);

    // text: | x2 | 6 | 0 | 1 |  3/8 | -1/4 | 3 |
    let r = row(&t, "x2");
    assert!(close(r.rhs, 3.0), "x2 rhs = {}", r.rhs);
    assert!(close(-r.coeffs[s1], 0.375), "x2/s1 a_ij = {}", -r.coeffs[s1]);
    assert!(close(-r.coeffs[s2], -0.25), "x2/s2 a_ij = {}", -r.coeffs[s2]);
}

/// `c_B` is reconstructed from the constraint map, not read from the live
/// objective row (where basic symbols have been substituted out).
#[test]
fn c_b_is_reconstructed_not_zero() {
    let (solver, _, _, names) = setup();
    let t = solver.tableau(&names);
    let costed: Vec<f64> = t.rows.iter().map(|r| r.c_b).filter(|c| *c != 0.0).collect();
    let mut costed = costed;
    costed.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(costed, vec![6.0, 7.0],
               "expected the two error variables to carry costs 6 and 7");
}

/// Column and row ordering must defeat `HashMap` iteration order, which is
/// randomised per process. Without this, output reshuffles between runs and
/// step-by-step comparison is worthless.
#[test]
fn rendering_is_deterministic() {
    let (a, _, _, na) = setup();
    let (b, _, _, nb) = setup();
    let ra = a.tableau(&na).render(&RenderOpts::textbook());
    let rb = b.tableau(&nb).render(&RenderOpts::textbook());
    assert_eq!(ra, rb);
    // And stable across repeated renders of one snapshot.
    let t = a.tableau(&na);
    assert_eq!(t.render(&RenderOpts::dictionary()), t.render(&RenderOpts::dictionary()));
}

/// Every basic row must satisfy `rhs == constant + sum(coeffs[j] * value_j)`
/// with nonbasic values at zero, i.e. the dictionary invariant reduces to
/// `rhs == value of the basic symbol`.
#[test]
fn basic_values_match_get_value() {
    let (solver, x1, x2, names) = setup();
    let t = solver.tableau(&names);
    for (v, n) in [(x1, "x1"), (x2, "x2")].iter() {
        let r = row(&t, n);
        assert!(close(r.rhs, solver.get_value(*v)),
                "{}: tableau rhs = {} but get_value = {}", n, r.rhs, solver.get_value(*v));
    }
}


/// The solver's carried objective constant is not the objective value once an
/// edit variable has been suggested: `suggest_value` encodes the new
/// right-hand side by shifting row constants directly, with no matching
/// correction to the objective row. Nothing in the solver reads that constant,
/// so this is harmless to the algorithm - but it must not be displayed as `z`.
#[test]
fn carried_constant_drifts_across_suggest_value() {
    let w = Variable::new();
    let l = Variable::new();
    let r = Variable::new();
    let mut names = HashMap::new();
    names.insert(w, "w".to_string());
    names.insert(l, "l".to_string());
    names.insert(r, "r".to_string());

    let mut solver = Solver::new();
    solver.add_constraints(&[
        w |GE(REQUIRED)| 0.0,
        l |EQ(REQUIRED)| 0.0,
        r |LE(REQUIRED)| w,
        r - l |EQ(1.0)| 50.0,
    ]).unwrap();
    solver.add_edit_variable(w, 1_000_000.0).unwrap();

    // Before any edit, the two agree.
    let t = solver.tableau(&names);
    assert!(close(t.carried_constant_drift(), 0.0));

    solver.suggest_value(w, 300.0).unwrap();
    let t = solver.tableau(&names);
    assert!(!close(t.carried_constant_drift(), 0.0),
            "expected the carried constant to drift, got {}", t.carried_constant_drift());
    // At width 300 the preferred width of 50 is satisfiable, so z is 0.
    assert!(close(t.objective_value(), 0.0), "z = {}", t.objective_value());

    // Squeeze below the preferred width: the WEAK error absorbs the shortfall.
    solver.suggest_value(w, 20.0).unwrap();
    let t = solver.tableau(&names);
    assert!(close(t.objective_value(), 30.0), "z = {}", t.objective_value());

    // The per-column identity survives all of it - it involves only
    // coefficients, never the drifting constant.
    let zj = t.zj();
    for (j, c) in t.columns.iter().enumerate() {
        assert!(close(c.cost - zj[j], t.objective.reduced[j]),
                "column {}: c_j - Z_j = {} but reduced = {}",
                c.name, c.cost - zj[j], t.objective.reduced[j]);
    }
}

/// A dummy column may carry a negative reduced cost in an optimal tableau:
/// `get_entering_symbol` skips dummies, because a dummy marks a REQUIRED
/// equality and must stay nonbasic at zero.
#[test]
fn dummy_columns_are_exempt_from_optimality() {
    let (solver, _, _, names) = setup();
    let t = solver.tableau(&names);
    assert!(t.is_optimal(), "solver returned from add_constraint, so it is optimal");
    if let Some(j) = t.entering_candidate() {
        panic!("entering candidate {} should not exist", t.columns[j].name);
    }
}
