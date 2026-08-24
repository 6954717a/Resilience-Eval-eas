# Resilience evaluation boundaries

The evaluation pipeline has three deliberately different lifecycles. They
must not share one large collector interface.

1. **Step monitors** (`evaluation.monitors`) own small online accumulators.
   A monitor implements `reset`, `update(step_data)`, and `get_summary` and is
   managed by `MonitorRegistry`. Inputs are routed by monitor name because a
   safety observation and a performance observation do not have the same
   schema.
2. **Episode collectors** (`evaluation.metrics.collection`) consume one
   immutable `EpisodeCollectionContext` after a rollout. Rebound and the
   runtime Stability diagnostic implement `collect_episode(context)` and are
   selected through `CollectorRegistry`.
3. **Dataset postprocessors** consume multiple rollouts. Stability
   neighborhood beta and Boundary/GE require clean references or a stress
   grid, so they are not registered as single-episode collectors.

`OnlineReboundTracker` remains an event-state machine (`reset(episode)`,
`observe`, `finalize`) rather than being disguised as a step monitor. Its
canonical transition adapter joins each executed planner decision with its
post-execution environment outcome; the collector consumes that transition
stream instead of reconstructing deltas from unrelated planner-info rows.
If a disturbance window reaches the episode tail without an evidence-backed
`t_r`, the tracker records a right-censored diagnostic interval. It is skipped
from formal `C_rec` and makes that episode result N/A instead of treating the
tail as a successful recovery.

Collectors receive copied monitor summaries, never live monitor objects.
Boundary/GE prefers the authoritative `unsafe_rate`; legacy CBF penalty rates
are treated as already containing collision penalties and are not added to
`collision_rate` a second time. Safety rows carry `safety_valid`,
`safety_scope`, and a missing reason. The current runner's collision scope is
only `agent_pair_collision`; it must not be described as general contact with
the scene, furniture, or obstacles. GE declares its required channels and
fails closed when the runtime scope does not cover them.

Formal Stability also fails closed unless the critic export declares a full
trajectory, selected transitions equal total transitions, transition indices
are contiguous, and usable feature coverage reaches the configured minimum
(1.0 by default). Sparse diagnostic exports are not StageBaseline input.
The standalone Stability result is the configured neighborhood supremum. GE
uses a separate beta-at-lambda table, so evidence from a higher stress level
cannot leak into a lower-lambda boundary cell.

The former single-episode Degradation monitor/collector has been removed. GE
owns degradation-curve diagnostics across the configured lambda grid; they are
not a fourth runtime metric.
