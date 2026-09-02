//! The two-box layout from the crate's module docs, with the tableau printed
//! after every step.
//!
//! This is the interesting one: it shows an *incremental* solver. There is no
//! "iteration 0, all-slack basis at the origin" — the tableau is already
//! optimal, and each `add_constraint` appends one row and re-optimises.
//! `suggest_value` then runs the *dual* simplex, which is why resizing a window
//! is cheap.
//!
//! Run with: cargo run --example two_box --features tableau
extern crate cassowary;

use std::collections::HashMap;

use cassowary::{Solver, Variable};
use cassowary::WeightedRelation::*;
use cassowary::strength::{WEAK, MEDIUM, STRONG, REQUIRED};
use cassowary::tableau::RenderOpts;

struct Element {
    left: Variable,
    right: Variable,
}

/// Phase 1 exit criterion: every external basic symbol's RHS is that variable's
/// value, so the tableau and `get_value` must agree at every step.
fn check(solver: &Solver, names: &HashMap<Variable, String>, vars: &[Variable], step: &str) {
    let t = solver.tableau(names);
    for v in vars {
        let name = &names[v];
        if let Some(r) = t.rows.iter().find(|r| &r.basis == name) {
            let got = solver.get_value(*v);
            assert!((r.rhs - got).abs() < 1e-9,
                    "{}: {} tableau rhs = {} but get_value = {}", step, name, r.rhs, got);
        }
    }
    // The per-column identity `c_j - Z_j == reduced[j]` is the robust invariant:
    // it holds after `add_constraint` and after `suggest_value` alike, because it
    // involves only coefficients. The *aggregate* form is not checked here - the
    // solver's carried objective constant drifts across edits.
    let zj = t.zj();
    for (j, c) in t.columns.iter().enumerate() {
        assert!(((c.cost - zj[j]) - t.objective.reduced[j]).abs() < 1e-6,
                "{}: column {}: c_j - Z_j = {} but reduced cost = {}",
                step, c.name, c.cost - zj[j], t.objective.reduced[j]);
    }
}

fn main() {
    let window_width = Variable::new();
    let box1 = Element { left: Variable::new(), right: Variable::new() };
    let box2 = Element { left: Variable::new(), right: Variable::new() };

    let mut names = HashMap::new();
    names.insert(window_width, "win.w".to_string());
    names.insert(box1.left, "b1.l".to_string());
    names.insert(box1.right, "b1.r".to_string());
    names.insert(box2.left, "b2.l".to_string());
    names.insert(box2.right, "b2.r".to_string());
    let all = [window_width, box1.left, box1.right, box2.left, box2.right];

    let opts = RenderOpts::dictionary();
    let mut solver = Solver::new();

    solver.add_constraints(&[
        window_width |GE(REQUIRED)| 0.0,
        box1.left |EQ(REQUIRED)| 0.0,
        box2.right |EQ(REQUIRED)| window_width,
        box2.left |GE(REQUIRED)| box1.right,
        box1.left |LE(REQUIRED)| box1.right,
        box2.left |LE(REQUIRED)| box2.right,
        box1.right - box1.left |EQ(WEAK)| 50.0,
        box2.right - box2.left |EQ(WEAK)| 100.0,
    ]).unwrap();
    check(&solver, &names, &all, "after constraints");
    println!("=== 1. all constraints added, window width still free ===\n");
    println!("{}", solver.tableau(&names).render(&opts));

    solver.add_edit_variable(window_width, STRONG).unwrap();
    solver.suggest_value(window_width, 300.0).unwrap();
    check(&solver, &names, &all, "width 300");
    println!("=== 2. width = 300 (roomy; both boxes get preferred widths) ===\n");
    println!("{}", solver.tableau(&names).render(&opts));

    solver.suggest_value(window_width, 75.0).unwrap();
    check(&solver, &names, &all, "width 75");
    println!("=== 3. width = 75 (too tight; a WEAK constraint must break) ===\n");
    println!("{}", solver.tableau(&names).render(&opts));

    solver.add_constraint(
        (box1.right - box1.left) / 50.0 |EQ(MEDIUM)| (box2.right - box2.left) / 100.0
    ).unwrap();
    check(&solver, &names, &all, "with ratio");
    println!("=== 4. plus a MEDIUM ratio constraint to control the degradation ===\n");
    println!("{}", solver.tableau(&names).render(&opts));

    for v in all.iter() {
        println!("{:>6} = {}", names[v], solver.get_value(*v));
    }
}
