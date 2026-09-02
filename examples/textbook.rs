//! Encodes the worked example from `example.md` and prints the tableau.
//!
//!   max 7*x1 + 6*x2   s.t.  2*x1 + 4*x2 <= 16,  3*x1 + 2*x2 <= 12,  x >= 0
//!   optimum: x1 = 2, x2 = 3, z = 32
//!
//! Cassowary has no user objective; it only ever minimises weighted constraint
//! violation. So the objective is encoded as a one-sided pull toward a target
//! far outside the feasible region: `x1 >= M` at strength 7 creates an error
//! variable costed 7, and minimising `7*(M - x1)` maximises `7*x1`.
//!
//! Run with: cargo run --example textbook --features tableau
extern crate cassowary;

use std::collections::HashMap;

use cassowary::{Solver, Variable};
use cassowary::WeightedRelation::*;
use cassowary::strength::REQUIRED;
use cassowary::tableau::RenderOpts;

fn main() {
    // A target far outside the feasible region (x1 <= 4, x2 <= 4).
    const M: f64 = 1000.0;

    let x1 = Variable::new();
    let x2 = Variable::new();

    let mut names = HashMap::new();
    names.insert(x1, "x1".to_string());
    names.insert(x2, "x2".to_string());

    let mut solver = Solver::new();

    // Constraint order fixes symbol ids, since `id_tick` is a counter:
    //   x1 -> 1, s(x1>=0) -> 2, x2 -> 3, s(x2>=0) -> 4,
    //   s5 = slack of `2x1+4x2<=16`  (the text's s1)
    //   s6 = slack of `3x1+2x2<=12`  (the text's s2)
    solver.add_constraints(&[
        x1 |GE(REQUIRED)| 0.0,
        x2 |GE(REQUIRED)| 0.0,
        2.0 * x1 + 4.0 * x2 |LE(REQUIRED)| 16.0,
        3.0 * x1 + 2.0 * x2 |LE(REQUIRED)| 12.0,
    ]).unwrap();

    println!("=== structural constraints only (no objective yet) ===\n");
    println!("{}", solver.tableau(&names).render(&RenderOpts::textbook()));

    // The objective, as costed error variables.
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();

    println!("=== with objective encoded (optimal) ===\n");
    let t = solver.tableau(&names);
    println!("{}", t.render(&RenderOpts::textbook()));

    println!("=== same state, solver's own dictionary form ===\n");
    println!("{}", t.render(&RenderOpts::dictionary()));

    println!("x1 = {}", solver.get_value(x1));
    println!("x2 = {}", solver.get_value(x2));
    println!("objective.value        = {}", t.objective_value());
    println!("sum(c_B(i) * rhs_i)    = {}", t.objective_value());
}
