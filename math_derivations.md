# Math Derivations: Territorial Belief on the Theory-of-Space Benchmark

This document gives the full formal derivation of the *territorial belief*
representation, its update rule, and the analytic relationships to the
stability and belief-inertia metrics in Zhang et al. (ICLR 2026, §5.1 / §5.3).
It is written as a research-quality derivation that can be lifted directly
into the LaTeX paper draft.

---

## 1 · POMDP setting (recap of the ToS framing)

The Theory-of-Space environment is a partially-observable Markov decision
process

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \Omega, T, O, c, H \rangle.
$$

* $\mathcal{S}$: world states, each a tuple $(R, \{(p_i, \phi_i, k_i)\}_{i=1}^{N}, p_a, \phi_a)$
  where $R$ is the room graph (multi-room, tree topology), $p_i \in \mathbb{Z}^2$
  is object $i$'s grid position, $\phi_i \in \{\mathrm{N,E,S,W}\}$ its facing,
  $k_i$ its room id, and $(p_a, \phi_a)$ the agent's pose.
* $\mathcal{A} = \{\textsc{Goto}(j),\,\textsc{Rotate}(\theta),\,\textsc{Observe},\,\textsc{Query}(j),\,\textsc{Term}\}$.
* $\Omega$: egocentric observations restricted to a 90° FOV; each visible
  object reports a discretised $(\text{bearing},\,\text{distance})$ pair
  and (if applicable) facing.
* Cost $c$: 1 for `Observe`, 2 for `Query`; 0 otherwise. Episode horizon
  $H = 20$ for active rollouts.

A spatial belief is then any function $B : \mathcal{H}_t \to \Delta(\mathcal{S})$
that maps the action–observation history $h_t = (o_0, a_0, \dots, a_{t-1}, o_t)$
to a distribution over world states. ToS evaluates the *modal estimate*
$\hat B_t = \arg\max_S B_t(S)$ rather than the full distribution, by asking the
agent to externalise it as a structured cognitive-map JSON
$\hat B_t \in \mathcal{J}$.

---

## 2 · Two cognitive-map representations

### 2.1 Flat LLM cogmap (the baseline ToS evaluates)

The ToS prompt (`vagen/.../prompts/cogmap_prompts.py`,
`COGMAP_INSTRUCTION_GLOBAL_ONLY`) elicits a flat JSON of the form

$$
\hat B^{\text{flat}}_t = \big\{ \text{name} \mapsto (p,\, \phi)\big\},
$$

with no structural unit between *names* and the global frame. Every object
is updated independently, and the model's internal belief over object pose
and the agent's own pose is exposed as a single dictionary.

### 2.2 Territorial cogmap (this work)

We introduce an intermediate latent unit, the **territory**, mirroring the
hippocampal/entorhinal place- and grid-cell organisation of biological
spatial memory. Formally,

$$
\boxed{\;
B^{\text{ter}}_t = \big( \{T_k\}_{k=1}^{K_t},\; A_t,\; \mathcal{O}_t,\; c_t \big),\;}
$$

where

* $T_k = (\mathcal{O}_k,\, D_k,\, v_k,\, \tau_k)$ is the $k$-th territory
  (one per visited room): its observed-object set $\mathcal{O}_k$, its
  door-cell set $D_k$, a visit count $v_k \in \mathbb{N}$, and the last
  turn the agent was inside it $\tau_k \in \{0, 1, \dots, t\}$.
* $A_t = (p_a,\, \phi_a,\, k_a)$ is the agent's pose plus its current
  territory id.
* $\mathcal{O}_t : \text{name} \to (k,\, p,\, \phi)$ is the global object
  table.
* $c_t : \text{name} \to \mathbb{N}_{\ge 1}$ is a per-object **confidence
  counter** that gates updates.

The flat cogmap is recovered as a marginalisation:

$$
\hat B^{\text{flat}}_t(\text{name}) = (\, p_{\mathcal{O}_t(\text{name})},\, \phi_{\mathcal{O}_t(\text{name})}\,),
$$

so the territorial representation is *strictly more expressive* than the
flat one (it carries $K + N$ extra integer scalars). Crucially, it is also
*structurally* different: updates are scoped at the territory level.

---

## 3 · The update rule

### 3.1 Observation lifting

An `Observe` action returns, for each visible object $i$, an ego-centric
$(b_i, d_i, \phi_i^{\text{obs}})$ triple — bearing in the 5-bin scheme
$\{\text{front-left, front-slight-left, front, front-slight-right, front-right}\}$,
distance in the 6-bin scheme
$\{\text{same, near, mid, slightly far, far, very far}\}$.

We lift this into a global candidate position by composing with the agent's
own pose:

$$
p_i^{\text{new}} \,=\, p_a + R(\phi_a)
\begin{pmatrix} d_i \sin b_i \\ d_i \cos b_i \end{pmatrix},
$$

where $R(\phi_a)$ is the rotation that maps the agent's local
forward direction to the global $+y$ axis. This introduces a single-cell
discretisation error $\le \tfrac{1}{2}$ unit per observation, which is the
*intrinsic noise floor* of any agent-pose-conditioned reconstruction.

### 3.2 The four-state object-update machine

Given an existing record $\mathcal{O}_t(i) = (k_i^{\text{old}}, p_i^{\text{old}}, \phi_i^{\text{old}})$
(or the empty case $i \notin \mathcal{O}_t$), we update as follows. Let
$\Delta_i = \|p_i^{\text{new}} - p_i^{\text{old}}\|_2$ and let $\epsilon$ be
a tolerance hyperparameter (we use $\epsilon = 1.5$, which is one
ego-bin worth of slack).

$$
\bigl(\mathcal{O}_{t+1}, c_{t+1}\bigr)(i) \;=\;
\begin{cases}
\bigl((k_a, p_i^{\text{new}}, \phi_i^{\text{new}}),\; 1\bigr) & i \notin \mathcal{O}_t \quad\textbf{(new)} \\[6pt]
\bigl(\mathcal{O}_t(i),\; c_t(i) + 1\bigr) & \Delta_i \le \epsilon \;\wedge\; k_a = k_i^{\text{old}} \quad\textbf{(reinforced)} \\[6pt]
\bigl(\mathcal{O}_t(i),\; c_t(i) - 1\bigr) & \bigl(\Delta_i > \epsilon \;\vee\; k_a \ne k_i^{\text{old}}\bigr) \;\wedge\; c_t(i) > 1 \quad\textbf{(buffered)} \\[6pt]
\bigl((k_a, p_i^{\text{new}}, \phi_i^{\text{new}}),\; 1\bigr) & \bigl(\Delta_i > \epsilon \;\vee\; k_a \ne k_i^{\text{old}}\bigr) \;\wedge\; c_t(i) \le 1 \quad\textbf{(overwritten)}
\end{cases}
$$

The agent state is updated unconditionally:

$$
A_{t+1} = (p_a^{\text{new}},\, \phi_a^{\text{new}},\, k_a^{\text{new}}),
\qquad v_{k_a^{\text{new}}} \mathrel{+}= 1,\qquad
\tau_{k_a^{\text{new}}} \mathrel{=} t + 1.
$$

This is a **state machine on the per-object confidence counter**:

```
       reinforced (c↑)
       ┌─────────────┐
       ▼             │
 ┌─────────────┐   ┌─┴──────────┐
 │  unobserved │─▶│   stored    │
 └─────────────┘   └─┬──────────┘
                     │ contradict, c>1
                     ▼
                ┌─────────────┐  contradict, c=1
                │  buffered   │──────────────────┐
                └─┬──────────┘                   ▼
                  │ consistent obs        ┌─────────────┐
                  └────────────────▶       │ overwritten │
                                          └─────────────┘
```

The defining property: **it takes two contradictions to overwrite a record
that has been seen at least twice consistently.** This single departure
from "naive overwrite" is responsible for the stability and inertia
properties we now derive.

---

## 4 · Stability analysis (ToS §5.1)

### 4.1 ToS metric definition

Stability in ToS is measured turn-wise: for each (object, turn) pair where
the prediction was previously correct (above a similarity threshold $\theta$),
score 1 if the next-turn prediction is no worse, else 0. Aggregated as
the empirical ratio.

Formally, with similarity $s(\hat p, p^*) = e^{-\|\hat p - p^*\|/L}$
(ToS uses $L \approx \tfrac{1}{2}$ room-diagonal),

$$
\textsc{Stab}(\hat B) = \frac{1}{|\mathcal{P}|} \sum_{(i,t) \in \mathcal{P}}
\mathbf{1}\bigl[s(\hat p_i^{(t+1)}, p_i^*) \ge s(\hat p_i^{(t)}, p_i^*)\bigr],
$$

over the index set $\mathcal{P} = \{(i,t)\,:\, s(\hat p_i^{(t)}, p_i^*) \ge \theta\}$.

### 4.2 Naive overwrite is unstable

**Proposition 1.** *Suppose observations are i.i.d. Gaussian:
$p_i^{\text{obs}} = p_i^* + \eta$, $\eta \sim \mathcal{N}(0, \sigma^2 I)$. A
flat-cogmap update that overwrites on every observation has*

$$
\mathbb{E}\bigl[\textsc{Stab}_{\text{flat}}\bigr] =
\Pr\!\left[\,\|p^{\text{new}} - p^*\|_2 \le \|p^{\text{old}} - p^*\|_2\,\right] = \tfrac{1}{2}.
$$

*Proof.* $p^{\text{old}}$ and $p^{\text{new}}$ are i.i.d. with the same
distribution, so the event "$p^{\text{new}}$ is closer to $p^*$" has
probability exactly $\tfrac{1}{2}$ by symmetry. ∎

This recovers the empirical 0.50–0.65 stability range that ToS reports for
GPT-5.2 / Claude-4.5-Sonnet in the vision world (their stability scores
sit between random and slightly-better-than-random, consistent with a
mostly-overwrite policy).

### 4.3 Buffered update is asymptotically stable

**Proposition 2.** *Let $T_{\text{buf}}$ be the territorial update with
buffer size $b \ge 2$ (i.e. require $b$ contradictions to overwrite). Let
$\mathrm{ECDF}_\sigma(\epsilon) = \Pr[\,\|\eta\| \le \epsilon\,]$ be the
ego-noise CDF at tolerance $\epsilon$. Then for any object that has been
observed $b+1$ times,*

$$
\Pr\bigl[\hat p_i^{(t+1)} \ne \hat p_i^{(t)}\bigr]
\;\le\;
\bigl(1 - \mathrm{ECDF}_\sigma(\epsilon)\bigr)^{b}.
$$

*Proof.* An overwrite occurs only after $b$ consecutive contradictions; each
contradiction has probability $1 - \mathrm{ECDF}_\sigma(\epsilon)$ under
i.i.d. observations; chain by independence. ∎

For $\epsilon = 1.5$ and $\sigma \approx 1$ (one ego bin), this gives
contradiction probability $\approx 0.13$, so a $b=2$ buffer pushes the
spurious-overwrite rate to $\approx 0.017$ — translating empirically to
$\textsc{Stab} \to 1.0$ as in our smoke test.

---

## 5 · Belief Inertia analysis (ToS §5.3)

### 5.1 ToS metric definition

After the false-belief perturbation that secretly relocates / reorients
$k = 4$ objects, ToS defines positional inertia as

$$
s_i^{\text{pos}} \;=\;
\underbrace{\frac{e_i^\top v_i}{\|e_i\|\,\|v_i\| + \varepsilon}}_{\cos\theta_i}
\;\cdot\;
\underbrace{\exp\!\left(-\frac{\|b_i^{\text{new}} - b_i^{\text{old}}\|^2}{2\sigma^2}\right)}_{w_i\,(\text{proximity weight})},
$$

where $b_i^{\text{old}}$ is the pre-shift belief, $b_i^{\text{new}}$ the
post-revision belief, $g_i^{\text{new}}$ the post-shift ground truth,
$v_i = b_i^{\text{old}} - g_i^{\text{new}}$ (the obsolete-prior offset),
and $e_i = b_i^{\text{new}} - g_i^{\text{new}}$ (the residual after revision).
A positive $s_i^{\text{pos}}$ means *the agent's update is pulled back
toward the obsolete location* — the empirical signature of inertia.

ToS reports vision GPT-5.2 at $s^{\text{pos}} \approx 0.347$ — *one third
of all perturbed objects show systematic pull-back.*

### 5.2 Why flat cogmap exhibits inertia

The LLM's free-form update is autoregressive: each new $b_i^{\text{new}}$
is conditioned on the entire chat history including $b_i^{\text{old}}$.
Empirically (Carlini et al. 2024, Wang et al. 2025) such updates exhibit
a *prior anchoring* phenomenon — the new prediction is a convex
combination $b_i^{\text{new}} \approx \alpha\,b_i^{\text{old}} + (1-\alpha)\,p_i^{\text{obs}}$
with $\alpha \approx 0.3$–0.7. Substituting:

$$
e_i = (1-\alpha)\,(p_i^{\text{obs}} - g_i^{\text{new}}) + \alpha\,v_i,
$$

so under low observation noise $\|p_i^{\text{obs}} - g_i^{\text{new}}\| \to 0$,
$e_i \to \alpha v_i$ and
$\cos\theta_i \to 1$, $w_i \to 1$, hence $s_i^{\text{pos}} \to 1$. **The
inertia metric is by construction bounded above by $\alpha$, the LLM's
prior weight.**

### 5.3 Territorial cogmap breaks inertia in two regimes

For an object inside the *same* territory after the perturbation, the
buffered rule absorbs the first contradiction (no update, $b_i^{\text{new}}
= b_i^{\text{old}}$, so $s_i^{\text{pos}} = 1$ on turn $t+1$ but
$\|e_i\| \to v_i$ and the inertia computation simply records the obsolete
prior verbatim). The *second* observation then triggers a clean overwrite
to $p_i^{\text{obs}}$, at which point

$$
e_i = p_i^{\text{obs}} - g_i^{\text{new}} \approx \eta,\qquad \|e_i\| \ll \|v_i\|,
$$

and $s_i^{\text{pos}} \to 0$. **The territorial inertia profile is therefore
*bimodal*: very high on the first conflicting observation, and near zero
after the second** — the natural diagnostic separator of "inertia" from
"detected change in progress".

For an object that *changed territory* after the perturbation (e.g. moved
through a doorway), territorial belief takes the territory mismatch as
*sufficient* evidence to overwrite immediately, side-stepping the
pixel-level autoregressive bias entirely.

---

## 6 · Familiarity signal (Day-3 stretch goal)

We expose

$$
f_k(t) = \exp\!\bigl(-\lambda\,(t - \tau_k)\bigr) \cdot \frac{\log(1 + v_k)}{\log V_{\max}},
$$

with $\lambda = 0.05$ (decay) and $V_{\max} = 20$ (saturation). Two intended
uses:

1. **Familiarity-weighted tolerance.** Replace the constant $\epsilon$ with
   $\epsilon_k(t) = \epsilon_0 \cdot (1 - f_k(t))$. Familiar territories have
   *tighter* tolerance, so any contradiction in a well-explored area is
   trusted faster — eliminating the "first-observation absorbed" property
   under high familiarity.
2. **Familiarity-aware exploration.** Bias the agent to revisit territories
   with low $f_k$, restoring information-gain-optimal exploration without
   needing an explicit AC-3 constraint store as in ToS's `Strategist` proxy.

Both are scoped as future work; only the read-only signal is exposed in
the current `_meta` field of the emitted cogmap.

---

## 7 · Summary of analytic claims

| Claim | Setting | Result |
|---|---|---|
| Flat overwrite is unstable | i.i.d. obs noise | $\mathbb{E}[\textsc{Stab}] = \tfrac{1}{2}$ |
| Buffered ($b=2$) is stable | i.i.d. obs noise | spurious overwrite $\le (1-\mathrm{CDF})^b$ |
| LLM inertia is autoregressive prior weight | autoregressive update | $s^{\text{pos}} \to \alpha$ |
| Territorial inertia is bimodal | buffered update | $s^{\text{pos}} \to 0$ after 2nd obs |
| Cross-territory change → instant overwrite | territory mismatch | $s^{\text{pos}} \to 0$ on first obs |

These are testable on the Theory-of-Space data: the smoke test trends
(stability $0.67 \to 1.00$, ori-correctness $0.67 \to 1.00$) are
consistent with Propositions 2 and §5.3 above. Falsification of any of
the inertia claims on the AutoDL-rendered run is itself a publishable
diagnostic finding.

---

## References (cited above)

(Full citations in `lit_review_condensed.md`.)

* Zhang et al., *Theory of Space*, ICLR 2026.
* Wang et al., *VAGEN*, NeurIPS 2025.
* O'Keefe & Dostrovsky, *Brain Research*, 1971.
* Hafting et al., *Nature*, 2005.
* Banino et al., *Nature*, 2018.
* Burt, *Journal of Mammalogy*, 1943.
* Powell & Mitchell, *J. Mammalogy*, 2012.
* Wimmer & Perner, *Cognition*, 1983.
* Knauff et al., *Spatial Belief Revision*, J. Cogn. Psych., 2013.
