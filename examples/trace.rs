//! Traces the solver's pivots, rather than only the state between calls.
//!
//! `Solver::tableau` shows endpoints. Everything interesting happens *inside*
//! `add_constraint`, which runs `create_row` -> `choose_subject` ->
//! `substitute` -> `optimise`, and `optimise` pivots to completion before
//! returning. `Solver::start_trace` records each of those pivots.
//!
//! Part 1 encodes the worked example from `example.md` (as `examples/textbook.rs`
//! does) and shows the primal simplex walking to the optimum.
//!
//! Part 2 does a `suggest_value` on an edit variable, which is the path a UI
//! layout engine like ratatui takes on every resize: shift the right-hand side,
//! then repair feasibility with *dual* pivots. No primal iteration at all.
//!
//! Run with: cargo run --example trace --features tableau
extern crate cassowary;

use std::collections::HashMap;

use cassowary::strength::{REQUIRED, STRONG, WEAK};
use cassowary::tableau::RenderOpts;
use cassowary::WeightedRelation::*;
use cassowary::{Solver, Variable};

fn main() {
    primal();
    println!("\n\n");
    dual();
}

/// The `example.md` LP, traced pivot by pivot.
fn primal() {
    // A target far outside the feasible region (x1 <= 4, x2 <= 4), so the
    // encoded objective pulls as hard as it can and never goes slack.
    const M: f64 = 1000.0;

    let x1 = Variable::new();
    let x2 = Variable::new();

    let mut names = HashMap::new();
    names.insert(x1, "x1".to_string());
    names.insert(x2, "x2".to_string());

    let mut solver = Solver::new();

    // Structural constraints first, so the tableau is feasible before the
    // objective is introduced. Symbol ids follow constraint order.
    solver
        .add_constraints(&[
            x1 |GE(REQUIRED)| 0.0,
            x2 |GE(REQUIRED)| 0.0,
            2.0 * x1 + 4.0 * x2 |LE(REQUIRED)| 16.0,
            3.0 * x1 + 2.0 * x2 |LE(REQUIRED)| 12.0,
        ])
        .unwrap();

    println!("################ part 1: primal simplex ################\n");
    println!(
        "Structural constraints are in place; x1 = {}, x2 = {}. Now the \n\
         objective goes in, as two costed error variables, and the trace \n\
         records every pivot `add_constraint` makes internally.\n",
        solver.get_value(x1),
        solver.get_value(x2)
    );

    // Record only the objective constraints, not the structural set-up.
    solver.start_trace(names.clone());
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    let trace = solver.stop_trace().unwrap();

    println!("{}", trace.render(&RenderOpts::textbook()));

    println!(
        "{} steps, {} of them pivots.",
        trace.len(),
        trace.pivots().len()
    );
    println!("x1 = {}, x2 = {}", solver.get_value(x1), solver.get_value(x2));
    println!(
        "(the text reaches x1 = 2, x2 = 3, z = 32 in two pivots; this encoding \n\
         carries extra dummy and slack columns, so it takes a few more)"
    );
}

/// A `suggest_value` resize, traced. This is the dual simplex path.
fn dual() {
    let window_width = Variable::new();
    let left = Variable::new();
    let right = Variable::new();
    let mid = Variable::new();

    let mut names = HashMap::new();
    names.insert(window_width, "win_w".to_string());
    names.insert(left, "left".to_string());
    names.insert(right, "right".to_string());
    names.insert(mid, "mid".to_string());

    let mut solver = Solver::new();

    // Two boxes side by side, splitting the window down the middle - the
    // shape of a ratatui horizontal layout.
    solver
        .add_constraints(&[
            window_width |GE(REQUIRED)| 0.0,
            left |EQ(REQUIRED)| 0.0,
            right |EQ(REQUIRED)| window_width,
            mid |GE(REQUIRED)| left,
            right |GE(REQUIRED)| mid,
            mid - left |EQ(WEAK)| right - mid,
        ])
        .unwrap();

    solver.add_edit_variable(window_width, STRONG).unwrap();

    println!("################ part 2: dual simplex (a resize) ################\n");
    println!(
        "`suggest_value` does not re-solve. It shifts the right-hand side in \n\
         place, which can drive basic variables negative, and then repairs \n\
         feasibility with dual pivots: the leaving row is chosen first (the \n\
         infeasible one), and the entering column by a ratio test along that \n\
         row. This is why a UI resize is cheap.\n"
    );

    solver.start_trace(names.clone());
    solver.suggest_value(window_width, 300.0).unwrap();
    let trace = solver.stop_trace().unwrap();

    println!("{}", trace.render(&RenderOpts::dictionary()));

    println!(
        "after suggest_value(300): win_w = {}, left = {}, mid = {}, right = {}",
        solver.get_value(window_width),
        solver.get_value(left),
        solver.get_value(mid),
        solver.get_value(right)
    );
    println!(
        "{} steps, {} of them pivots.",
        trace.len(),
        trace.pivots().len()
    );
}
