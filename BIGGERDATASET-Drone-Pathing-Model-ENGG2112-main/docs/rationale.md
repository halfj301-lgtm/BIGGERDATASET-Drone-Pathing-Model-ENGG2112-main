# Design Rationale

A running log of the architectural and reward-design decisions made for the
ENGG2112 PPO drone pathfinding project. The intent is to give the final report
a defensible "we tried X, here is why" trail rather than a sequence of
unexplained constant changes.

Each section is dated and tied to a training run (`models/run_*/`) so that
the corresponding metrics can be cross-referenced.

---

## Project review — 2026-05-11

A sense-check of the project mid-way through. Recorded so the eventual writeup
can quote the diagnosis and trace each subsequent change to it.

### Where things stood

- Architecture: flat MLP `[256, 256]` over a 450-dim observation
  (9 scalars + flattened 21×21 elevation patch).
- Reward: shaping + step_cost + altitude penalty + valley bonus + time
  penalty + terminal bonuses.
- Curriculum stage: `fixed_set` (10 endpoint pairs) at the point of the review;
  Phase 3 (`random`) not yet attempted.
- Best evaluation (run `run_20260511_180552`, vs A*):
  - distance −3.4% (RL slightly shorter)
  - elevation gain **+196%**
  - shared A* cost **+131%**
  - goal reached: yes
  - inference time 50× faster than A* search

### Diagnosis

Two structural problems were identified, both of which would block the
Phase 3 generalisation target rather than just produce a sub-optimal Phase 2
policy:

1. **The observation is spatially blind.** Flattening the 21×21 patch into an
   MLP forces the network to relearn 2D adjacency from raw indices. For a
   2000×1500 grid where typical A* paths are ~1800 cells long, the agent has
   no mechanism to plan around terrain — only to react to the 84 m around its
   feet plus the global goal vector. That is enough to learn "head south for
   these 10 fixed pairs" but not enough to learn a generalisable terrain
   policy.
2. **The reward is teaching the wrong thing.** Per-step accounting at a
   mid-elevation cell shows the valley bonus (≈ +0.30) is roughly 6× the
   shaping signal (+0.05) and ~6× the per-step cost penalty. The training
   curve shows rolling episode length dropping from ~5100 to ~3700 then
   climbing back to ~4700 while reward stays positive — the textbook
   signature of a per-step bonus exploit. The agent is paid more for
   loitering in valleys than for reaching the goal.

The 2-D path overlay confirms both diagnoses: the RL path runs in nearly a
straight line from start to goal, ignoring the valley A\* follows. It cannot
see the mountain it is about to climb (problem 1) and the loss function does
not strongly disincentivise climbing (problem 2 — the cost term is dwarfed
by the valley bonus going the other way).

### Final-submission targets

Confirmed with the user before starting the rework:

- Headline result is **Phase 3 random-endpoint generalisation**, not
  fixed-set performance.
- Model performance and writeup quality weighted roughly equally — the plan
  must produce a clean ablation table, not just a single best run.
- PPO + 8-action discrete action space is locked. Action-space and
  algorithm changes are off the table.
- Many 1–4 h training runs are feasible on the RTX 4080. Architectural
  changes are on the table.

### Plan (5 steps, 1 baseline + 4 ablations + 1 headline run)

| Run | Step | Change | Purpose in writeup |
|---|---|---|---|
| Baseline (v0) | — | existing `run_20260511_180552` | starting point |
| A | 2 | MLP + valley bonus disabled | "reward rebalance" ablation |
| B | 3 | + CNN feature extractor | "spatial encoder" ablation |
| C | 4 | + multi-scale patch | "multi-scale context" ablation |
| D | 5 | warm-start from C, ENDPOINT_MODE=random | **generalisation result** |

The runs share everything except the one variable changed at their step, so
each row of the ablation table isolates one effect. Run D is the only
random-endpoint run; everything before it stays on the 10-pair fixed set so
the comparison is apples-to-apples.

Decisions sections for steps 1–5 are appended to this file as each step
lands.

---

## Step 1 — Success-rate logging — 2026-05-11

### Decision

Add `info["outcome"]` to every terminal/truncated step return from
`DronePathEnv`, with values `"success"`, `"timeout"`, `"oob"`, or `"nan"`.
Extend `TrainingPlotCallback` with a third subplot showing rolling
success rate (window = 100 episodes).

### Why this came first

Every subsequent step needs to be diagnosed against a success metric, not
just episode length and reward. Without it, a long-but-successful episode
looks identical to a hovering-exploit episode, and the reward-rebalance
ablation (step 2) cannot be evaluated honestly. This is pure
instrumentation — no observation, action, or reward change — so it cannot
contaminate any downstream comparison.

### Why outcome rather than a boolean success flag

There are three distinct failure modes (oob, nan, timeout) and lumping
them together hides useful signal: a high oob rate means the agent is
exploring aggressively into invalid cells; a high timeout rate means it's
hovering or getting lost; a high nan rate means it's hitting holes in the
DEM. Recording the discrete outcome leaves room to plot the failure-mode
breakdown later if needed, at no extra cost up front.

### Files touched

- `rl/environment.py` — `step()` info dict on all four terminal paths.
- `rl/train.py` — `TrainingPlotCallback.__init__`, `_on_step`, `_save_plot`.

### What was deliberately not touched

- `_ep_successes` field already existed and was unused. Left in place to
  avoid breaking anything; subsequent cleanup can remove it.
- Reward, observation, action space, and all hyperparameters are unchanged
  by this step. Existing checkpoints remain loadable.

---

## Step 2 — Reward rebalance — 2026-05-11

### Decision

Set `W_VALLEY = 0.0` in `rl/config.py`. Leave `W_ALTITUDE = 0.3` untouched.
Leave the valley-bonus code path in `reward.py` intact so the term can be
toggled back on for an ablation if needed.

### Why disable the valley bonus

Per-step accounting (described in the project review above) showed the
valley term contributing ≈+0.30 at full credit, against a +0.05 shaping
signal and ~−0.05 step-cost penalty. With the time penalty at −0.05, a
step *away* from the goal but inside a valley nets out to roughly +0.20
reward — a positive return on hovering. The training-curve evidence
(episode length rebounding from 3700 to 4700 while reward stayed
positive) was consistent with this exploit.

This is a one-constant change, so it produces a clean before/after pair
of training runs for the ablation table.

### Why not also zero W_ALTITUDE

`W_ALTITUDE` penalises absolute elevation, which is a different signal
from the slope cost already in `step_cost`. It is not redundant in the
same way the valley bonus was. It may still be misaligned with A* (A*
does not include an altitude term), but removing it is a second,
separate ablation question — keeping it in Run A isolates the valley-
bonus effect cleanly. If Run A's path is still elevation-heavy compared
to A*, `W_ALTITUDE` is the next constant to ablate, but not yet.

### Why leave the code path in `reward.py`

The valley term is one line. Deleting it would make the ablation
"valley on vs valley off" a multi-file change. Keeping it in and toggled
by the config constant means an ablation is a one-line edit and one
fresh run.

### Files touched

- `rl/config.py` — `W_VALLEY` set to 0.0 with a comment pointing back here.

### What was deliberately not touched

- All other reward constants (`W_DIST`, `W_ELEV`, `W_POWER`, `COST_SCALE`,
  `SHAPING_SCALE`, `GOAL_BONUS`, `FAILURE_PENALTY`, `TIME_PENALTY`,
  `W_ALTITUDE`) unchanged.
- `reward.py` unchanged — the valley term still executes, it just
  contributes zero.

### Run A — what to look for

After this step, kick off a full training run on `ENDPOINT_MODE =
"fixed_set"` with the unchanged MLP `[256, 256]` architecture. The
acceptance criteria for the rebalance hypothesis:

- Rolling success rate climbs past 60% by ~2 M steps and continues to
  improve (instead of stagnating around the long-episode plateau).
- Rolling episode length decreases monotonically rather than rebounding
  after an initial dip.
- Final RL-vs-A* cost ratio improves on the baseline +131%.

If episode length still climbs after 2 M steps, the diagnosis was wrong
and `W_ALTITUDE` becomes the next suspect before moving to architecture
changes.

---

## Step 3 — CNN feature extractor — 2026-05-11

### Decision

Add a custom `BaseFeaturesExtractor` (`DronePatchCNN` in `rl/policy.py`)
that splits the flat observation vector into the 9 scalar features and
the 21×21 elevation patch, runs the patch through three convolutional
layers (1→16→32→32 channels, last with stride 2), then concatenates the
result with the scalars and projects to a 128-dim feature vector. PPO's
standard MLP `[256, 256]` policy/value heads sit on top of this. The
observation space, env, and reward function are unchanged.

### Why a CNN over the patch

The flat MLP has to learn 2D adjacency from raw indices. For terrain,
this is wasteful and unlikely to generalise: a convolution at the same
relative offset within the patch always sees the same neighbourhood
structure, which is exactly the inductive bias we want. This is the
single change with the largest expected impact on Phase 3 generalisation.

### Why scalars bypass the CNN

The 9 scalars are global state (position, goal direction, distance,
accumulated cost) — they have no 2D structure and shouldn't be reshaped
into a 1×1 image. Concatenating them after the CNN trunk is the
standard pattern for hybrid state + image observations.

### Why obs space unchanged

Keeping the env's flat-vector output means existing scripts
(`evaluate.py`, `compare.py`, `view3d.py`) continue to work, and the
A*-vs-RL comparison runs against the same env at evaluation time. The
extractor unpacks the vector internally — `BaseFeaturesExtractor` is
exactly the SB3 hook for this.

### Why 128-dim features

A reasonable midpoint. Larger gives the policy more headroom but
inflates the policy head's first linear layer. 128 keeps total
extractor params at ~510 k (492 k of that is the post-CNN linear
projection from 3 881 → 128) which is fine on a 4080.

### Files touched

- `rl/policy.py` — new file (~90 lines including docstring and shape
  guard).
- `rl/train.py` — one import, four-key `policy_kwargs`, one comment.

### What was deliberately not touched

- `DronePathEnv` and the observation vector layout. The CNN reads from
  the same flat vector the env already produces.
- `PPO_HIDDEN = [256, 256]`. The MLP downstream is preserved so the
  ablation is "feature extractor only."
- Reward function and reward constants. Same reward as Run A.

### Run B — what to look for

Train with `ENDPOINT_MODE = "fixed_set"`, all reward weights identical
to Run A, full 10 M steps. Compare against Run A's success rate, episode
length, and final RL-vs-A* cost gap. Acceptance criterion: success rate
reliably ≥ 85 % by 4 M steps and final RL cost gap closer to A* than
Run A. If Run B is no better than Run A, either the CNN is too small or
the patch is too local to matter (in which case Step 4's multi-scale
addition is doubly important).

### Compatibility note

Pre-Step-3 checkpoints (`run 1/`, `run_20260511_*`) cannot be loaded by
the new policy — the `features_extractor` state-dict shape differs. The
old runs are kept on disk as reference but no longer warm-startable.
This is the point at which the project becomes "post-CNN" for all
future training.

---

## Step 4 — Multi-scale patch — 2026-05-11

### Decision

Add a second elevation patch (`PATCH_SIZE_COARSE × PATCH_SIZE_COARSE`,
sampled at `COARSE_STRIDE` cells between samples) to the observation.
The CNN feature extractor now takes a 2-channel image (fine and coarse
stacked on the channel dimension) instead of 1-channel.

Initial values:
- `PATCH_SIZE_COARSE = 21`
- `COARSE_STRIDE = 8`

At `DEM_STEP = 4` m/cell, this gives the coarse patch a vision radius of
`(21//2) × 8 × 4 m = 320 m` from the agent (≈ 640 m square). The fine
patch covers a 40 m radius. The two scales together cover the
neighbourhood where most local terrain decisions matter, plus enough of
the surrounding map to *see* an upcoming ridge or valley before
committing.

### Why this is needed at all

The fine-only observation gives the agent visibility comparable to
walking in fog with a 40 m torch. Phase 3 random endpoints range over
~8 km of terrain — the policy needs a wider view to plan around
obstacles rather than react to them.

### Why coarse_stride = 8, not 4 or 16

- Stride 4 doubles vision to ~160 m but still misses most regional
  terrain — only marginal improvement over fine.
- Stride 16 gives ~1.3 km of vision but the coarse sample becomes very
  blurry — single pixels span 64 m and may straddle features. Useful
  if Run C fails to generalise, but starts over-aliased.
- Stride 8 is the midpoint that gives substantial new vision without
  losing all resolution. A defensible default; can be tuned later if
  Run D doesn't generalise.

### Why same `PATCH_SIZE` for both, stacked as channels

Two design alternatives were considered:
1. Two separate CNN branches (one per scale) then concatenate → more
   expressive but ~2× CNN params and harder to write up cleanly.
2. Stack on channel dim → simpler, the same conv kernels learn to mix
   the two scales, fewer params.

Chose option 2 because the first conv layer's filters can already
combine information across channels — this is how RGB CNNs work. The
"channels-as-scales" pattern is well-established in remote sensing CNN
work. If ablations show that the coarse channel is being ignored, we'd
revisit option 1 — but that's a follow-up, not a blocker.

### Why share normalisation with the fine patch

The coarse patch reuses `self._elev_min` and `self._elev_range` from
the same DEM. This keeps the two channels on a comparable scale (the
CNN can compare a fine pixel to a coarse pixel directly), and it means
there is only one normalisation parameter set to remember at evaluation
time.

### Files touched

- `rl/config.py` — added `PATCH_SIZE_COARSE` and `COARSE_STRIDE`.
- `rl/environment.py` — extended docstring, added coarse-patch params
  to `__init__`, added `_get_coarse_patch` method, updated `_get_obs`
  and `obs_dim`.
- `rl/policy.py` — first conv now takes 2 channels, forward splits and
  stacks both patches before the CNN, obs-dim guard updated.

### Compatibility note

Run B's checkpoints (CNN, single-scale) are not loadable by the new
extractor — the first conv layer's input channel count differs and the
obs vector length changed. Run C trains from scratch.

### Run C — what to look for

Train fresh with `ENDPOINT_MODE = "fixed_set"`, same reward as Runs A
and B. Acceptance criteria:

- Success rate at convergence at least as high as Run B (the coarse
  channel should not hurt on the fixed set).
- RL-vs-A* cost gap noticeably tighter than Run B — the coarse view
  should let the agent route around hills it previously climbed
  straight over.
- If Run C is not better than Run B on the fixed set, the coarse
  channel is not contributing useful information — try doubling
  `COARSE_STRIDE` to 16 and re-running, or revisit option 1
  (separate-branch CNN).

If Run C beats Run B, this is the architecture that goes into Run D
for the Phase 3 random-endpoint result.

### Performance note

`_get_coarse_patch` is a Python double loop (441 iterations per obs).
That is roughly 5 µs per obs build at 12 envs × millions of steps —
expected to add a small but not dominant overhead. If profiling shows
it as a hot spot, the loop is trivially replaceable with `np.ix_`
indexing or `np.take_along_axis`. Optimise only if measured to matter.

---

## Step 5 — Phase 3 warm-start (`--resume`) — 2026-05-11

### Decision

Add a `--resume PATH` CLI flag to `rl/train.py`. When set, training
loads an existing `ppo_final.zip` via `PPO.load`, lowers the learning
rate to `RESUME_LR = 5e-5`, and continues training. When
`ENDPOINT_MODE = "random"` is set without `--resume`, training prints a
warning but does not block (so a deliberate from-scratch random-mode
experiment is still possible).

### Why warm-start Phase 3 at all

Phase 3 random endpoints are dramatically harder than Phase 2's
ten-pair fixed set. Trying to learn random-endpoint navigation from
random weights wastes compute on rediscovering basic skills the
fixed-set policy already has (goal-seeking, avoiding NaN cells, basic
shaping signal). The standard curriculum-learning recipe is to lock in
the easy task, then continue training on the hard one — which is the
"warm-start from Run C" plan.

### Why LR = 5e-5

CLAUDE.md flags `PPO_LR ≤ 1e-4` and `PPO_CLIP_RANGE ≤ 0.1` as the safe
band for fine-tuning an existing policy without catastrophic forgetting.
Half of the lower bound (5e-5) is a conservative starting point — gives
the policy enough learning rate to adapt to the wider state distribution
of random endpoints without overwriting what it already knows. If
Run D's success rate stalls, this is the first dial to nudge (up to
1e-4) before any architectural change.

### Why the warning rather than a hard block

A reasonable use case (debugging, ablation) is to *deliberately* run
random mode from scratch to confirm warm-start is actually helping.
Blocking the user from doing this would be paternalistic. The warning
captures the most common mistake (forgetting `--resume` for Run D)
without removing the option.

### Why refresh `lr_schedule`

SB3 caches a `lr_schedule` callable derived from the learning rate at
model-construction time. Setting `model.learning_rate = RESUME_LR`
alone is insufficient — the cached schedule still returns the old LR
during optimisation. Calling `model.lr_schedule = get_schedule_fn(RESUME_LR)`
ensures the optimiser actually uses the new rate. This is a known SB3
gotcha; without it the resume "looked" lowered but trained as usual.

### Files touched

- `rl/train.py` — `import argparse`, module-level `RESUME_LR`, argparse
  block at start of `main()`, an `if args.resume / else` split around
  the PPO construction, warning print when random mode is used without
  `--resume`.

### Run D — the headline result

```
python -m rl.train --resume models/run_<C>/ppo_final.zip
```

With `ENDPOINT_MODE = "random"` set in `config.py`. Suggested
`TOTAL_TIMESTEPS = 5_000_000` for the warm-start phase (random is
harder per step but starts from a competent policy, so half the steps
should still produce a usable agent).

Acceptance criteria for the final writeup:
- Success rate on random endpoints converges above ~50% (any
  generalisation at all is a result on this curriculum).
- A handful of qualitative comparisons on unseen seed pairs
  (use `rl/compare.py`) showing the policy adapts to terrain it has
  not seen.
- Final RL-vs-A* cost gap on random endpoints, expected to be wider
  than on the fixed set but within the same ballpark — the headline
  number for the report.

If success rate stays at zero through Run D, the next investigation
points are (in order): coarse-patch stride, learning-rate, exploration
entropy coefficient (currently SB3 default).

---

## Implementation status

All five planned code-side steps are in place. The remaining work is
training-time experimentation:

| Run | Step | Code state | Action |
|---|---|---|---|
| A | 2 | ready | run with `W_VALLEY = 0.0`, fixed_set, 10 M steps |
| B | 3 | ready | run with CNN extractor, otherwise identical to A |
| C | 4 | ready | run with multi-scale CNN, otherwise identical to B |
| D | 5 | ready | `--resume` from C with `ENDPOINT_MODE = "random"` |

The user owns the actual `python -m rl.train` invocations and the
ablation numbers that come out of them. This document gets one final
update per run with the numbers and any deviations from the plan.

---

## Step 6 — Reward rebalance, take 2 — 2026-05-11

### Failure observed

First Run C (multi-scale CNN, fixed_set, `W_VALLEY=0`, all other reward
weights at original values) reached 8.5 M of 10 M steps with:

- Success rate: **0.0 %** (no episode ever reached the goal in ~1700+
  episodes)
- Episode length: pinned at the 5000-step cap
- Reward: stuck around −700 to −1000, no monotonic improvement after
  ~1 M steps

This is the failure mode Step 1's success-rate panel was added to catch.
Without it the run would have looked like "reward going up slowly,
maybe converging" — Step 1 paid for itself before any other change
shipped.

### Diagnosis

Step 2 disabled the valley bonus to stop a hovering exploit. The
diagnosis was correct, but I missed that `W_VALLEY = +0.3` was also the
only consistently *positive* per-step reward term in the function.
Removing it without rebalancing left every per-step component
negative for a typical state:

| term | value at mid-elev cell moving toward goal |
|---|---|
| shaping (Δdist × 0.05) | +0.05 |
| step_cost / 100 | −0.05 |
| altitude penalty (0.3 × 0.5) | −0.15 |
| valley bonus | 0.00 |
| time penalty | −0.05 |
| **net per step** | **−0.20** |

Moving toward the goal had negative reward. The locally-optimal policy
was "do nothing useful and time out." The agent never reached the goal,
so the +50 terminal bonus was never sampled — it had no anchor to learn
from.

Curve evidence matched the math: net-negative per step × ~5000 steps =
~−1000 reward per episode, which is what the plot showed.

### Decision

Three constants changed:

| constant | old | new | reason |
|---|---|---|---|
| `SHAPING_SCALE` | 0.05 | **0.2** | Make the directional signal large enough to dominate the per-step noise. Toward-goal shaping is now +0.20 per step, away-from-goal is −0.20. |
| `TIME_PENALTY` | −0.05 | **−0.01** | Still discourages hovering, but doesn't drown the shaping signal. The previous value was 5× the new shaping signal. |
| `W_ALTITUDE` | 0.3 | **0.0** | This term was paid every step regardless of direction. It was the largest single negative drag on the reward and adds nothing the slope cost in step_cost doesn't already cover (slope cost penalises climbing; altitude penalty just biases the whole map toward low cells). Also better aligns the training reward with the A\* cost (A\* has no altitude term). |

### Expected per-step reward after change

| direction | net |
|---|---|
| toward goal | **+0.14** |
| away from goal | **−0.26** |

Now there is a clear sign-difference between "good" and "bad" actions
and the cumulative shaping over an episode (≈ +400 for a 2000-cell
distance reduction) is the same order of magnitude as the goal bonus.
The reward landscape is no longer a uniform negative pit.

### Why this isn't reintroducing the valley exploit

The valley bonus was a per-step bonus tied to *being at a local
elevation minimum*, regardless of progress. Shaping is tied to *getting
closer to the goal*, which by definition isn't hoverable — you can't
keep collecting shaping reward without moving toward the goal. Time
penalty is still negative so standing still also loses reward.

### Files touched

- `rl/config.py` — three constants changed, comments left in place
  pointing back to this section.

### What was deliberately not touched

- `W_DIST`, `W_ELEV`, `W_POWER`, `COST_SCALE`, `GOAL_BONUS`,
  `FAILURE_PENALTY`, `W_VALLEY` unchanged.
- CNN architecture, multi-scale patch, success-rate logging, resume
  flag — all unchanged. The architecture is fine; the reward was the
  bug.

### Acceptance criteria for the re-run

- Rolling success rate visibly non-zero by 1 M steps.
- Episode length should drop below the cap within 2 M steps.
- Reward should cross zero by the time success rate is ≥ 30 %.

If success rate is still 0 % at 2 M steps after this change, the
remaining suspects are (in order): exploration entropy too low,
goal_bonus too small relative to FAILURE_PENALTY, max_steps too tight.
None of those are likely given the new per-step numbers, but they are
the next dials to try.

### Writeup angle

This is one of the more honest stories the report can tell: an
ablation chain where the first change (kill valley bonus) was
qualitatively correct but quantitatively destructive, and the failure
mode was caught immediately because Step 1's instrumentation went in
first. The fix is principled (signed per-step reward analysis), not
"we kept tuning until it worked." Worth a paragraph in the methodology
section.

---

## Step 7 — Disable CNN extractor + add exploration bonus — 2026-05-11

### Failure observed

After Step 6's reward rebalance, training restarted with the multi-scale
CNN architecture. At 3.9 M / 10 M steps:

- Success rate: **still 0.0 %**
- Episode length: pinned at the 5000-step cap (no early termination)
- Reward improving slowly from −1000 to −500, but converging to a
  non-goal-reaching local optimum

Per-step accounting on the new reward: at ~−0.10 reward per step over
5000 steps (= −500 episode reward), shaping contribution averages
~−0.04 per step, which means the agent moves ~0.2 cells *away* from
the goal on average. The policy is anti-correlated with goal direction
despite a +0.20 toward-goal shaping signal.

### Diagnosis

The CNN feature extractor was the prime suspect. At initialisation:
- CNN trunk has 3 872 noisy features (random conv weights)
- 9 scalar goal-direction features bypass the CNN
- They are concatenated and projected via Linear(3 881 → 128)
- Ratio at init: 3 872 noisy CNN features : 9 informative scalars
  → ~430 : 1 noise-to-signal in the projection's input

Result: the policy head can't see the goal direction through the CNN
noise until the conv layers learn something useful — and they can't
learn anything useful without a working policy producing on-policy
data. Classic cold-start CNN deadlock. The flat MLP didn't have this
problem because the 9 scalars sat at 9 / 450 ≈ 2 % of the input and
went straight to the policy head.

### Decision

Two minimal changes, both reversible:

1. **Disable CNN feature extractor.** Comment out
   `features_extractor_class` and `features_extractor_kwargs` in
   `train.py`. SB3 falls back to the default `MlpExtractor` over the
   flat 891-dim observation. The multi-scale obs from Step 4 is
   preserved — only the upstream CNN is removed.
2. **Add entropy bonus** (`ent_coef=0.01`). Encourages action diversity
   so the policy doesn't collapse to a near-deterministic wander early
   in training. SB3 default is 0.0, which is fine for problems where
   the gradient signal alone provides exploration — but in a sparse-
   goal navigation task with a fresh policy, that's not enough.

Both changes are one-liners. The CNN code path is preserved (commented,
not deleted) so re-enabling is trivial after the reward + obs are
proven to work with the MLP.

### Why this is the right kind of retreat

This is not abandoning the architecture work. It is isolating the
variables:

- If the MLP + multi-scale obs + new reward + entropy now learns,
  we know the *reward fix* (Step 6) and the *multi-scale obs* (Step 4)
  are sound. The CNN was the bug.
- If even the MLP can't learn, the reward or obs needs more debugging
  before any architectural change matters.

Either way, the next data point is interpretable. Continuing to debug
with three changed-at-once components (reward + multi-scale + CNN) was
producing only the conclusion "something is broken."

### Files touched

- `rl/train.py` — commented out `features_extractor_*` keys; added
  `ent_coef=0.01` to the `PPO(...)` constructor.

### What was deliberately not touched

- `rl/policy.py` — left intact. The CNN extractor class still exists;
  it just isn't wired in.
- Reward constants (Step 6 values stand).
- Multi-scale patch in env (Step 4 stands — env still produces 891-dim
  obs).
- Success-rate logging (Step 1 stands).
- `--resume` flag (Step 5 stands).

### Acceptance criteria for the next run

- Rolling success rate visibly non-zero by 1 M steps.
- Episode length falls below the 5000 cap by 2 M steps.
- Reward crosses zero by the time success rate reaches ~30 %.

### What to do based on the result

| Run D outcome | Interpretation | Next move |
|---|---|---|
| Success rate climbs | Reward + multi-scale obs are sound; CNN was the bug | Re-enable CNN with smaller `features_dim` (e.g. 32) and/or smaller conv channels |
| Success rate still 0 % | Either reward or multi-scale obs is also broken | Try reverting `PATCH_SIZE_COARSE`-channel from obs vector (revert step 4) and rerunning MLP-only on the original 450-dim obs |

### Writeup angle

The honest story in the report: we made three changes (reward rebalance
+ CNN + multi-scale patch) and shipped them as a single architectural
update. When training failed, instrumentation from Step 1 caught it
immediately, but isolating the cause required falling back to a known-
working architecture. Lesson worth recording: "ablate one variable at
a time" was the right plan, and we partially deviated from it — the
project review correctly identified all three problems, but the user-
preferred ordering (apply all the fixes together) coupled them. The
recovery (this step) restored the ablation discipline.






