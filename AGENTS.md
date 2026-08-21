# Repository Agent Contract

## Mission

Own stock-model research for this repository. Reproduce, compare and evaluate predictive architectures or signals with point-in-time, versioned data and explicit out-of-sample evidence. Code or a fitted model alone is not a verified research result.

## Canonical authority

- Consume market/fundamental data from their owning canonical sources where possible; do not create duplicate finance-wide data authorities merely for model experiments.
- Preserve universe, sample/split, feature availability time, target definition, benchmark, transaction-cost assumptions, model/config version and run provenance.
- Keep observed inputs, features, predictions, evaluation results and investment interpretation distinct.

## Autonomous execution

1. Inspect current `main`, README, open Issues/PRs, experiment/data manifests, models, workflows/tests and result artifacts.
2. Continue one canonical research workline before adding another architecture, dataset, branch or Issue.
3. Prefer completion of a reproducible OOS experiment, leakage/baseline correction, falsification of a hypothesis, or deletion of superseded experiment/code paths.
4. Predeclare the decision/claim, split, benchmark and metric before interpreting a model result.
5. Run focused deterministic/model evaluation checks and bind results to the exact code/data revision before merge.
6. Stop at the fixed point; do not sweep architectures or hyperparameters without a bounded evidence question.

## Branch lifecycle

- Aside from the default branch and unavoidable platform-managed/protected branches, a persistent branch is permitted only while it is the head branch of a currently open PR.
- Creating a work branch creates an obligation to open or reuse its canonical PR immediately; do not use branches as backlog, continuation state, backup, archive, or evidence storage.
- After a PR is merged or closed, delete its head branch after verifying PR/main state. A branch with no open PR is an orphan and must be deleted.
- Before and after work, compare repository branches with open PR heads. Do not report cleanup/fixed point while an orphan task branch remains.
- If the available tool cannot delete a branch, record that as a tooling blocker and do not claim cleanup complete. Never create another orphan branch as a workaround.

## Merge and release are separate

### PR merge conditions

A PR may merge when the repository-local research contract is correct on the exact head revision: point-in-time data/splits and benchmark definitions are fixed, deterministic/evaluation tests pass, result artifacts are reproducible where affected, and no unresolved review or correctness blocker remains.

A future market observation, production deployment, live inference, realized return, or end-user adoption is **not** a merge condition unless the PR specifically changes the release mechanism and pre-merge validation belongs to that bounded change.

### Research/model release conditions

Release is a separate post-merge decision. Treat a research/model result as released only after the merged `main` revision is read back and the release artifact/surface in scope is actually verified, including exact data/model revision, persisted evaluation artifact, published model/API/UI when applicable, deployment identity, and rollback/rebuild path.

A merged PR does not prove production predictive performance. A release/data/runtime blocker may block release without invalidating a correctly merged repository change. Report merge and release independently.

## Boundaries

- In-sample fit is not out-of-sample evidence; one seed is not robustness when stochastic variation matters.
- Do not infer missing data, costs, market impact or production performance.
- Never execute trades, rebalance accounts or change financial-account settings.
- Unexecuted experiments, CI and realized market outcomes remain unverified.

## Completion report

Report research status Before -> After, exact data/split/model/benchmark result, Issue/PR/commit/check artifact, then report `merged` and `released` separately with direct evidence for each. Include branch cleanup state, superseded complexity removed and the remaining blocker.