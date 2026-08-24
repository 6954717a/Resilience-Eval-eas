# Perturbation dose, realization, and response

Perturbations are controlled interventions, not outcome labels. The runtime
keeps three quantities separate:

1. **Requested dose** is the configured `PerturbationSpec.intensity` in
   `[0, 1]`.
2. **Realized dose** is computed from the change that was actually observed.
3. **Response** is the downstream change in trajectory, success, Rebound, or
   another evaluation metric.

Response is never used to define realized dose. This avoids putting model
performance on both sides of a stress-response calculation.

## Runtime contract

`PerturbationSpec` rejects unknown kinds and non-finite/out-of-range
intensities. An intensity of zero is an exact no-op. `EvaluationRunner`
performs prompt rewriting once and clears the legacy planner hook so the same
instruction cannot be rewritten again inside `LLMPlanner`.
Each `run_instruction` call starts a new injector rollout generation: repeated
calls inside that generation stay idempotent, while a retry that rebuilds the
WorldGraph must reapply and re-audit the intervention.

The backward-compatible methods return the perturbed value or an audit dict.
New code should prefer the explicit APIs:

```python
rewritten, audit = injector.rewrite_instruction_with_audit(instruction, episode)
audit = injector.apply_world_perturbation_with_audit(env_interface, episode)
```

Each `PerturbationAudit` contains `requested`, `realized`, `valid`, `reason`,
`components`, hashes, seed, and episode id. The runner also preserves the
episode audit in `info["perturbation_audit"]` and emits flat
`perturbation_requested`, `perturbation_realized`, `perturbation_valid`,
`perturbation_reason`, and stable-JSON `perturbation_components` fields for the
episode CSV. A positive-dose world mutation is invalid unless every target
world graph reports an observable planner-visible change.

## Family-specific realization

Prompt doses use the number of actually applied rewrite/filler operations and
also report deterministic text-edit and token-growth signals. For a fixed
instruction and seed, operation count is non-decreasing with requested dose.

`irrelevant_perturbation` reports distractor count, count/max count, and, when
`WorldGraph.get_world_descr()` is available, the planner-visible added-token
ratio. Its current headline realization (`count / maximum_count`) is an
**in-family dose only**; it is not numerically comparable to surface rewrite,
filler injection, or state corruption. Missing world-description measurement
or zero added planner-visible tokens invalidates the realization even if a
Python attribute was changed.

`room_shift` and `holding_toggle` require explicit readable/writable adapters
and are not part of the default grid. `object_state_toggle` is an
episode-grounded planner-belief corruption: it selects a real object-state
evaluation proposition, resolves that exact simulator handle/semantic alias in
every agent graph (including Furniture), and toggles the observed boolean in
the WorldGraph snapshot. It never treats geometric predicates such as
`is_on_floor` as boolean object states. Shared graph instances are mutated only
once; graph views that share the same Entity are also deduplicated. Multi-graph
changes are atomic with verified rollback. The current adapter requires full
observation; partial-observation runs fail closed rather than silently switch
to another visible proposition. Unsupported or unverifiable mutations fail
closed and must not enter a valid stress cell.
These interventions are currently binary: any successful toggle realizes dose
`1.0`, even when the requested grid value is fractional. They therefore
support validity/effect analysis, but not a smooth within-family dose-response
curve until multi-level adapters are implemented.

## Optional LLM candidate calibration

`LLMCandidateCalibrator` is an offline, standalone helper. It creates no client
and makes no network request by default. A caller may inject a completion
function that returns strict JSON to:

- propose semantics-preserving candidates; and
- reject candidates that appear to change task meaning.

Symbolic invariants such as required entities remain authoritative. The LLM
cannot override them, and it never supplies `realized` stress. Candidate text,
task-contract checks, deterministic change features, model identity, and
semantic-review output should be cached in an immutable bank before rollout.
Runtime evaluation should load a frozen candidate by id rather than call an
endpoint.

## Pairing and artifact identity

`derive_run_seed(experiment_seed, repetition_seed)` produces a paired execution
seed independent of perturbation family: the same repetition seed should be
used for clean and all matched perturbations. `spec_artifact_key(spec)` includes
the exact floating-point intensity and `extras` payload in a digest, avoiding
the collisions caused by two-decimal path formatting. Experiment runners must
use these helpers when they create worker paths/job ids and record the derived
run seed alongside the perturbation seed.

## Known limitations

- Hand-authored synonym/filler banks provide only a few discrete dose levels.
- A non-decreasing requested dose may map to the same realized dose; analysis
  should deduplicate or explicitly retain plateaus.
- Semantic equivalence still needs contract-aware checks and a small human
  validation sample for paper-facing perturbation banks.
- Family-specific realized doses require empirical calibration before any
  cross-family comparison.
- Stability sensitivity divides response by positive, valid realized dose;
  missing/invalid realization evidence fails closed instead of falling back to
  requested intensity.
- WorldGraph state corruption is a one-shot epistemic shock exposed to the
  initial planner call before the first environment step. A later simulator
  perception update may repair it; this recovery is part of the intervention
  semantics. It is not evidence that the physical simulator state changed or
  that a rollout-long overlay was maintained.
