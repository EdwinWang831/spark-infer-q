# E19 — nonlinear confidence-head capacity comparison

2026-10-22. The user authorized the next stage after the E17 completion path and an approximately 11K-parameter MLP comparison were proposed. E17 sources, inputs, selection rules and execution order remain frozen. E19 is a separate finite continuation using the same resident worker, after E17 terminates successfully (including a valid negative offline result).

## Question and fixed models

Does replacing the linear F1 conditional predictor with a small nonlinear model improve verification decisions enough to pay for its CPU inference cost?

The model is **137 → 64 → 32 → 7**, with tanh hidden activations. It has 11,143 network parameters plus 14 independent Platt calibration parameters, **11,157 total**, versus 980 for the F1 linear model. The 137 features, fixed 2560×16 hidden projection, seven conditional-label contract, full-seven DFlash generation, target weights, arithmetic and action set remain unchanged. This tests the combined nonlinear structure/shared representation/capacity change; it does not isolate parameter count alone.

Two candidates only: ordinary F1 MLP and regret-weighted F1 MLP. One fixed initialization seed 20261022. CPU float64, two threads, LBFGS80 with strong-Wolfe line search, L2=.005 on the sum of squared network weights. The objective sums separately averaged conditional-depth BCE losses. Only examples whose previous cumulative event is true contribute to the next depth. Regret weights are normalized separately within each eligible depth. Sparse or single-class depths use smoothed constant priors. Independent unweighted positive-slope Platt calibration follows the existing linear procedure. Normalization uses training rows only, with the same clipping and minimum scale.

## Same data; selection before test access

1. Reconstruct exactly the E17 old-train plus external-train row order and regret weights. Verify E17's dataset SHA map, train/calibration row counts, train/calibration group separation and weight summary. No new target labels, checkpoint downloads or target/draft fitting.
2. Reuse E17's training/calibration split, costs and fixed-width control. Old test/OOD rows never train or determine weights. The E19 fit stage does not open E17 new test collection/evaluation files.
3. Fit both MLPs once. Warm and remeasure both new models, both corresponding frozen linear F1 controls and the E17 calibration-selected control on the same calibration inputs. Nine rotated/reversed timing passes determine median scalar inference cost. The scalar Python backend, including normalization and probability calibration, is also the online backend.
4. Pick one MLP by calibration replay policy tokens/ms, including its measured inference cost. Seal both model artifacts, the selected candidate and SHA before opening test labels in E19.
5. Evaluate both predeclared MLPs on E17's 90 external test prompts/72 roots, keeping the calibration selection fixed. Report cumulative Brier errors, actions, grouped throughput estimates, capture and matched linear comparisons. **This is a prespecified comparison reusing E17's held-out split, not a newly acquired external test.** Do not tune architecture, seed or hyperparameters using these test results. The launch seal records whether E17 test files/results already existed.

## Promotion gate and online controls

The selected MLP must satisfy the E17 fixed-control gate: capture≥.30, positive grouped-bootstrap lower bounds for gain and capture, different actions, and positive English/Chinese gain lower bounds. Additionally, the grouped gain lower bounds must be positive against both its corresponding frozen F1 linear control (same ordinary/weighted training) and E17's selected model. All inference costs are included. Negative results end normally without online promotion or test-driven reselection.

If eligible, use the same 20 preselected E17 test roots (8 English/8 Chinese/4 mixed), four arms and three balanced repetitions: selected MLP, corresponding F1 linear, E17 selected model, and true target-only. That gives **240 timed runs and 80 independent committed-logit/live-state oracles**. Duplicate linear control identities, if any, are retained and reported. Draft/cache construction, updates, feature extraction, CPU transfer, prediction, scheduling, target verification and commit are timed. Prompt construction, state cloning and oracle comparisons are outside formal decode timing. All timed outputs and final states must match the serial oracle. No new native TensorFold ranking or default replacement is implied.

## Execution and evidence

`scripts/e19_pipeline.py` freezes source SHA and the contract, waits for E17's terminal success/valid negative state, then queues only one job at a time:

- `061-e19-smoke`: model and chain guard tests under the actual resident process's inference context.
- `062-e19-fit`: CPU-only MLP fitting and calibration-only selection.
- `063-e19-evaluation`: held-out evaluation, artifact `e19-evaluate.json`.
- `064-e19-online`: only after the new gate passes.

The resident PID and load_count=1 are checked throughout. No worker restart, model reload, automatic retry, original-source edits or concurrent experiment queue is introduced. Any source mismatch/job failure stops the chain and retains the failing record. A standalone Spark CPU preflight passed all ten model/chain tests on two separate CPU cores; the official serialized smoke still repeats them before real fitting. The preflight loaded no target/draft model and made no CUDA calls.

Raw data stays in ignored storage. `scripts/e19_export.py` exports explicit progress/negative/completed states, verifies selected-model/fit/evaluation/online bindings and sanitizes worker identifiers. Offline replay is never presented as measured online TPS. A separate future source/workload test is needed for a broad generalization claim.
