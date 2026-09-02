//! Step tracing: the invariants each recorded pivot must satisfy, and the
//! guard that recording does not perturb the algorithm.
//!
//! The pivot *sequence* is not reproducible across processes -
//! `get_entering_symbol` takes the first negative-cost symbol out of a
//! `HashMap` - so nothing here asserts a particular sequence. Every assertion
//! is a property that must hold of whatever sequence a run produces.
#![cfg(feature = "tableau")]

extern crate cassowary;

use std::collections::HashMap;

use cassowary::strength::{REQUIRED, STRONG, WEAK};
use cassowary::tableau::{RenderOpts, SymbolKind, Tableau, TraceEvent};
use cassowary::WeightedRelation::*;
use cassowary::{Solver, Variable};

const M: f64 = 1000.0;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// The `example.md` LP: `max 7*x1 + 6*x2` encoded as costed error variables.
fn textbook() -> (Solver, Variable, Variable, HashMap<Variable, String>) {
    let x1 = Variable::new();
    let x2 = Variable::new();
    let mut names = HashMap::new();
    names.insert(x1, "x1".to_string());
    names.insert(x2, "x2".to_string());

    let mut solver = Solver::new();
    solver
        .add_constraints(&[
            x1 |GE(REQUIRED)| 0.0,
            x2 |GE(REQUIRED)| 0.0,
            2.0 * x1 + 4.0 * x2 |LE(REQUIRED)| 16.0,
            3.0 * x1 + 2.0 * x2 |LE(REQUIRED)| 12.0,
        ])
        .unwrap();
    (solver, x1, x2, names)
}

/// A two-box layout with an edit variable, the shape a UI toolkit resizes.
fn layout() -> (Solver, Vec<Variable>, HashMap<Variable, String>) {
    let win = Variable::new();
    let left = Variable::new();
    let right = Variable::new();
    let mid = Variable::new();

    let mut names = HashMap::new();
    names.insert(win, "win".to_string());
    names.insert(left, "left".to_string());
    names.insert(right, "right".to_string());
    names.insert(mid, "mid".to_string());

    let mut solver = Solver::new();
    solver
        .add_constraints(&[
            win |GE(REQUIRED)| 0.0,
            left |EQ(REQUIRED)| 0.0,
            right |EQ(REQUIRED)| win,
            mid |GE(REQUIRED)| left,
            right |GE(REQUIRED)| mid,
            mid - left |EQ(WEAK)| right - mid,
        ])
        .unwrap();
    solver.add_edit_variable(win, STRONG).unwrap();
    (solver, vec![win, left, right, mid], names)
}

// ---------------------------------------------------------------------------
// The guard: instrumentation must not change the algorithm
// ---------------------------------------------------------------------------

#[test]
fn tracing_does_not_change_the_solution() {
    let (mut a, x1a, x2a, _) = textbook();
    a.add_constraint(x1a |GE(7.0)| M).unwrap();
    a.add_constraint(x2a |GE(6.0)| M).unwrap();

    let (mut b, x1b, x2b, names) = textbook();
    b.start_trace(names);
    b.add_constraint(x1b |GE(7.0)| M).unwrap();
    b.add_constraint(x2b |GE(6.0)| M).unwrap();

    assert!(close(a.get_value(x1a), b.get_value(x1b)));
    assert!(close(a.get_value(x2a), b.get_value(x2b)));
    assert!(close(b.get_value(x1b), 2.0));
    assert!(close(b.get_value(x2b), 3.0));
}

#[test]
fn tracing_does_not_change_a_resize() {
    let (mut a, va, _) = layout();
    a.suggest_value(va[0], 300.0).unwrap();

    let (mut b, vb, names) = layout();
    b.start_trace(names);
    b.suggest_value(vb[0], 300.0).unwrap();

    for i in 0..va.len() {
        assert!(
            close(a.get_value(va[i]), b.get_value(vb[i])),
            "variable {} diverged: {} vs {}",
            i,
            a.get_value(va[i]),
            b.get_value(vb[i])
        );
    }
    assert!(close(b.get_value(vb[0]), 300.0));
    assert!(close(b.get_value(vb[3]), 150.0));
}

#[test]
fn not_tracing_by_default() {
    let (mut solver, x1, x2, _) = textbook();
    assert!(!solver.is_tracing());
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    assert!(!solver.is_tracing());
    assert!(solver.stop_trace().is_none());
    assert!(solver.take_trace().is_empty());
}

// ---------------------------------------------------------------------------
// Recording lifecycle
// ---------------------------------------------------------------------------

#[test]
fn trace_records_the_pivots_add_constraint_runs_internally() {
    let (mut solver, x1, x2, names) = textbook();
    solver.start_trace(names);
    assert!(solver.is_tracing());
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    let trace = solver.stop_trace().unwrap();

    assert!(!solver.is_tracing());
    // Two constraints, each recording a `SubjectChosen` and an `Optimal`, plus
    // at least one pivot in between - which is the whole point: those pivots
    // are invisible to `Solver::tableau`.
    assert!(trace.len() >= 5, "only {} steps", trace.len());
    assert!(!trace.pivots().is_empty(), "no pivots recorded");

    let subjects = trace
        .steps
        .iter()
        .filter(|s| match s.event {
            TraceEvent::SubjectChosen { .. } => true,
            _ => false,
        })
        .count();
    assert_eq!(subjects, 2, "one SubjectChosen per add_constraint");

    // Optimality is where `optimise` stops, so the last step must be one.
    match trace.steps.last().unwrap().event {
        TraceEvent::Optimal { .. } => {}
        ref e => panic!("trace ends on {:?}, not Optimal", e),
    }
}

#[test]
fn take_trace_drains_and_keeps_recording() {
    let (mut solver, x1, x2, names) = textbook();
    solver.start_trace(names);

    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    let first = solver.take_trace();
    assert!(!first.is_empty());
    assert!(solver.is_tracing(), "take_trace must not stop recording");

    // The drain is real: this second batch covers only the second constraint.
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    let second = solver.take_trace();
    assert!(!second.is_empty());
    assert_eq!(solver.take_trace().len(), 0, "second drain should be empty");
}

#[test]
fn start_trace_discards_earlier_steps() {
    let (mut solver, x1, x2, names) = textbook();
    solver.start_trace(names.clone());
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.start_trace(names);
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    let trace = solver.stop_trace().unwrap();
    let subjects = trace
        .steps
        .iter()
        .filter(|s| match s.event {
            TraceEvent::SubjectChosen { .. } => true,
            _ => false,
        })
        .count();
    assert_eq!(subjects, 1, "steps from before the restart survived");
}

// ---------------------------------------------------------------------------
// Primal pivot invariants
// ---------------------------------------------------------------------------

/// Every primal pivot must satisfy the rules `optimise` pivots by:
/// the entering column has a negative reduced cost and is not a dummy; the
/// leaving row is restricted (non-External) with a negative coefficient in
/// that column; and it attains the minimum ratio.
#[test]
fn primal_pivots_obey_the_entering_and_ratio_rules() {
    let (mut solver, x1, x2, names) = textbook();
    solver.start_trace(names);
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    let trace = solver.stop_trace().unwrap();

    let mut checked = 0;
    for step in &trace.steps {
        if let TraceEvent::Pivot { .. } = step.event {
            let t: &Tableau = &step.tableau;
            let c = t.pivot_col.expect("pivot step without a marked column");
            let r = t.pivot_row.expect("pivot step without a marked row");

            assert!(
                t.objective.reduced[c] < 0.0,
                "entering column {} has reduced cost {}, not negative",
                t.columns[c].name,
                t.objective.reduced[c]
            );
            assert!(
                t.columns[c].kind != SymbolKind::Dummy,
                "a dummy column entered the basis"
            );
            assert!(
                t.rows[r].kind != SymbolKind::External,
                "an external row left the basis"
            );

            // In dictionary form the leaving coefficient is negative; the
            // ratio is -rhs/coeff, and no eligible row beats it.
            let pivot = t.rows[r].coeffs[c];
            assert!(pivot < 0.0, "pivot element {} is not negative", pivot);
            assert_eq!(Some(pivot), t.pivot_element());

            let best = -t.rows[r].rhs / pivot;
            for other in &t.rows {
                if other.kind == SymbolKind::External {
                    continue;
                }
                let coeff = other.coeffs[c];
                if coeff < 0.0 {
                    let ratio = -other.rhs / coeff;
                    assert!(
                        ratio >= best - 1e-9,
                        "{} has ratio {} beating the chosen {}",
                        other.basis,
                        ratio,
                        best
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no primal pivots to check");
}

/// A pivot step is the state *before* the pivot, so the entering symbol must
/// still be a column and the leaving symbol still be in the basis.
#[test]
fn pivot_snapshots_are_pre_pivot() {
    let (mut solver, x1, x2, names) = textbook();
    solver.start_trace(names);
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    let trace = solver.stop_trace().unwrap();

    let mut checked = 0;
    for step in &trace.steps {
        if let TraceEvent::Pivot {
            ref entering,
            ref leaving,
            ..
        } = step.event
        {
            let t = &step.tableau;
            let c = t.pivot_col.unwrap();
            let r = t.pivot_row.unwrap();
            assert_eq!(&t.columns[c].name, entering);
            assert_eq!(&t.rows[r].basis, leaving);
            // Neither has swapped yet.
            assert!(t.rows.iter().all(|row| &row.basis != entering));
            assert!(t.columns.iter().all(|col| &col.name != leaving));
            checked += 1;
        }
    }
    assert!(checked > 0);
}

/// Consecutive snapshots must be linked by the pivot the earlier one marks:
/// after it, the entering symbol is basic and the leaving symbol is not.
#[test]
fn successive_steps_apply_the_marked_pivot() {
    let (mut solver, x1, x2, names) = textbook();
    solver.start_trace(names);
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    let trace = solver.stop_trace().unwrap();

    let mut checked = 0;
    for i in 0..trace.steps.len() - 1 {
        let (entering, leaving) = match trace.steps[i].event {
            TraceEvent::Pivot {
                ref entering,
                ref leaving,
                ..
            }
            | TraceEvent::DualPivot {
                ref entering,
                ref leaving,
            } => (entering.clone(), leaving.clone()),
            _ => continue,
        };
        let next = &trace.steps[i + 1].tableau;
        assert!(
            next.rows.iter().any(|r| r.basis == entering),
            "{} did not enter the basis by the next step",
            entering
        );
        assert!(
            next.rows.iter().all(|r| r.basis != leaving),
            "{} is still basic after the pivot that removed it",
            leaving
        );
        checked += 1;
    }
    assert!(checked > 0);
}

// ---------------------------------------------------------------------------
// Dual pivot invariants
// ---------------------------------------------------------------------------

/// `suggest_value` shifts the right-hand side and repairs feasibility with
/// dual pivots: the leaving row is the infeasible one (negative rhs), and the
/// entering column minimises `reduced_cost / coefficient` over columns with a
/// *positive* coefficient in that row.
#[test]
fn dual_pivots_obey_the_leaving_and_dual_ratio_rules() {
    let (mut solver, vars, names) = layout();
    solver.start_trace(names);
    solver.suggest_value(vars[0], 300.0).unwrap();
    let trace = solver.stop_trace().unwrap();

    let mut checked = 0;
    for step in &trace.steps {
        if let TraceEvent::DualPivot { .. } = step.event {
            let t: &Tableau = &step.tableau;
            let c = t.pivot_col.expect("dual pivot without a marked column");
            let r = t.pivot_row.expect("dual pivot without a marked row");

            assert!(
                t.rows[r].rhs < 0.0,
                "leaving row {} has rhs {}, which is not infeasible",
                t.rows[r].basis,
                t.rows[r].rhs
            );
            assert!(
                t.rows[r].coeffs[c] > 0.0,
                "entering coefficient {} is not positive",
                t.rows[r].coeffs[c]
            );
            assert!(t.columns[c].kind != SymbolKind::Dummy);

            let ratios = t.dual_ratios.as_ref().expect("dual ratios not computed");
            let best = ratios[c].expect("entering column has no dual ratio");
            for j in 0..t.columns.len() {
                if let Some(other) = ratios[j] {
                    assert!(
                        other >= best - 1e-9,
                        "column {} has dual ratio {} beating the chosen {}",
                        t.columns[j].name,
                        other,
                        best
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no dual pivots recorded");
}

#[test]
fn suggest_value_records_the_shift_then_dual_pivots_then_feasible() {
    let (mut solver, vars, names) = layout();
    solver.start_trace(names);
    solver.suggest_value(vars[0], 300.0).unwrap();
    let trace = solver.stop_trace().unwrap();

    match trace.steps.first().unwrap().event {
        TraceEvent::ValueSuggested {
            ref variable,
            value,
        } => {
            assert_eq!(variable, "win");
            assert!(close(value, 300.0));
        }
        ref e => panic!("trace starts on {:?}, not ValueSuggested", e),
    }
    match trace.steps.last().unwrap().event {
        TraceEvent::Feasible => {}
        ref e => panic!("trace ends on {:?}, not Feasible", e),
    }
    // The shift left at least one row infeasible - that is what the dual
    // pivots then repair - and none remain by the end.
    assert!(trace.steps[0].tableau.rows.iter().any(|r| r.infeasible));
    assert!(trace
        .steps
        .last()
        .unwrap()
        .tableau
        .rows
        .iter()
        .all(|r| !r.infeasible));
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

#[test]
fn pivot_marks_appear_in_the_rendering() {
    let (mut solver, x1, x2, names) = textbook();
    solver.start_trace(names);
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    let trace = solver.stop_trace().unwrap();

    let step = trace
        .pivots()
        .into_iter()
        .next()
        .expect("no pivot to render");
    let out = step.render(&RenderOpts::textbook());
    let t = &step.tableau;

    assert!(
        out.contains(&format!("{}*", t.columns[t.pivot_col.unwrap()].name)),
        "entering column not marked:\n{}",
        out
    );
    assert!(
        out.contains(&format!("{} <", t.rows[t.pivot_row.unwrap()].basis)),
        "leaving row not marked:\n{}",
        out
    );
    assert!(out.contains("pivot element"), "no pivot legend:\n{}", out);
    assert!(out.contains("state *before* that pivot"));
}

#[test]
fn trace_rendering_is_deterministic_and_numbers_every_step() {
    let (mut solver, x1, x2, names) = textbook();
    solver.start_trace(names);
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    let trace = solver.stop_trace().unwrap();

    let a = trace.render(&RenderOpts::textbook());
    let b = trace.render(&RenderOpts::textbook());
    assert_eq!(a, b, "rendering leaked HashMap iteration order");

    for i in 1..=trace.len() {
        assert!(
            a.contains(&format!("--- step {} of {}:", i, trace.len())),
            "step {} missing from the rendering",
            i
        );
    }
}

/// A snapshot with no pending pivot must not print pivot marks.
#[test]
fn non_pivot_steps_carry_no_pivot_marks() {
    let (mut solver, x1, x2, names) = textbook();
    solver.start_trace(names);
    solver.add_constraint(x1 |GE(7.0)| M).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();
    let trace = solver.stop_trace().unwrap();

    for step in &trace.steps {
        if !step.event.is_pivot() {
            assert!(step.tableau.pivot_col.is_none());
            assert!(step.tableau.pivot_row.is_none());
            assert!(step.tableau.pivot_element().is_none());
            assert!(!step.render(&RenderOpts::dictionary()).contains("pivot element"));
        }
    }
}

// ---------------------------------------------------------------------------
// Artificial variables
// ---------------------------------------------------------------------------

/// Two lower bounds on the same variable force `add_with_artificial_variable`.
///
/// `create_row` flips a row with a negative constant so that it is positive
/// again, which usually leaves the marker slack with a negative coefficient
/// and so a usable subject. Stacking bounds in the *same* direction defeats
/// that: after `a >= 10` is basic, the row for `a >= 20` reduces to slacks
/// only, with the marker positive - no subject, so phase I runs.
fn stacked_bounds() -> (Solver, Variable, Variable, HashMap<Variable, String>) {
    let a = Variable::new();
    let b = Variable::new();
    let mut names = HashMap::new();
    names.insert(a, "a".to_string());
    names.insert(b, "b".to_string());
    let mut solver = Solver::new();
    solver.add_constraint(a |GE(REQUIRED)| 10.0).unwrap();
    (solver, a, b, names)
}

/// An artificial variable is created with `SymbolType::Slack`, so it can only
/// be identified by the symbol the solver records while phase I runs. Check
/// the label is exact and that it is confined to the phase-I window.
#[test]
fn artificial_phase_is_labelled_and_bounded() {
    let (mut solver, a, b, names) = stacked_bounds();
    solver.start_trace(names);
    solver.add_constraint(a |GE(REQUIRED)| 20.0).unwrap();
    solver.add_constraint(b |GE(REQUIRED)| a).unwrap();
    let trace = solver.stop_trace().unwrap();

    let starts: Vec<&str> = trace
        .steps
        .iter()
        .filter_map(|s| match s.event {
            TraceEvent::ArtificialPhaseStart { ref artificial } => Some(artificial.as_str()),
            _ => None,
        })
        .collect();

    // Non-vacuity: if this system stops needing phase I the test is worthless,
    // so fail rather than pass silently.
    assert_eq!(starts.len(), 1, "expected exactly one phase I, got {:?}", starts);

    // The artificial is a `Slack` by type; only the recorded symbol identifies
    // it, and the snapshot must show it as `Artificial`, not as a slack.
    let start = trace
        .steps
        .iter()
        .find(|s| match s.event {
            TraceEvent::ArtificialPhaseStart { .. } => true,
            _ => false,
        })
        .unwrap();
    assert!(
        start
            .tableau
            .rows
            .iter()
            .any(|r| r.kind == SymbolKind::Artificial && r.basis == starts[0]),
        "artificial {} is not a basis row labelled Artificial; basis is {:?}",
        starts[0],
        start
            .tableau
            .rows
            .iter()
            .map(|r| (&r.basis, r.kind))
            .collect::<Vec<_>>()
    );

    // At least one pivot must have happened inside phase I - that is the work
    // `Solver::tableau` alone cannot see.
    assert!(
        trace.steps.iter().any(|s| match s.event {
            TraceEvent::Pivot { phase, .. } => phase == cassowary::tableau::Phase::One,
            _ => false,
        }),
        "no phase I pivot recorded"
    );

    // Every start is matched by an end, and no snapshot outside a phase-I
    // window carries an Artificial symbol.
    let mut depth = 0i32;
    for step in &trace.steps {
        match step.event {
            TraceEvent::ArtificialPhaseStart { .. } => depth += 1,
            TraceEvent::ArtificialPhaseEnd { success } => {
                assert!(success, "phase I failed on a satisfiable system");
                depth -= 1;
                assert!(depth >= 0, "unmatched ArtificialPhaseEnd");
                continue;
            }
            _ => {}
        }
        if depth == 0 {
            let t = &step.tableau;
            assert!(
                t.columns.iter().all(|c| c.kind != SymbolKind::Artificial)
                    && t.rows.iter().all(|r| r.kind != SymbolKind::Artificial),
                "an artificial symbol survived outside phase I, at {:?}",
                step.event
            );
        }
    }
    assert_eq!(depth, 0, "phase I never closed");

    // The tighter bound wins, and b is pushed up to meet a.
    assert!(close(solver.get_value(a), 20.0), "a = {}", solver.get_value(a));
    assert!(close(solver.get_value(b), 20.0), "b = {}", solver.get_value(b));
}

// ---------------------------------------------------------------------------
// Constraint removal
// ---------------------------------------------------------------------------

#[test]
fn remove_constraint_records_its_marker_and_reoptimises() {
    let (mut solver, x1, x2, names) = textbook();
    let objective = x1 |GE(7.0)| M;
    solver.add_constraint(objective.clone()).unwrap();
    solver.add_constraint(x2 |GE(6.0)| M).unwrap();

    solver.start_trace(names);
    solver.remove_constraint(&objective).unwrap();
    let trace = solver.stop_trace().unwrap();

    let markers = trace
        .steps
        .iter()
        .filter(|s| match s.event {
            TraceEvent::MarkerRemoved { .. } => true,
            _ => false,
        })
        .count();
    assert_eq!(markers, 1, "expected exactly one MarkerRemoved");
    match trace.steps.last().unwrap().event {
        TraceEvent::Optimal { .. } => {}
        ref e => panic!("removal did not end on Optimal, but {:?}", e),
    }
    // With the pull on x1 gone, only x2 is still pushed toward M.
    assert!(close(solver.get_value(x2), 4.0), "x2 = {}", solver.get_value(x2));
}



