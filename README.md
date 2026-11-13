# RealPDE Track 2 — Long-Term Test-Time Adaptation

Submission to the NeurIPS 2026 RealPDE [competition](https://realpdecompetition.github.io), Track 2: Long-Term
Test-Time adaptation (LTTTA) for real PIV measurements of flow around a
NACA4418 airfoil (Re 2,968–27,975, angle of attack 0–20°).

Development Phase leaderboard: **final_score = 80.854482**.

## Folder structure

```
realpde-track2-submission/
├── submission.py
├── README.md
├── LICENSE
└── .gitignore
```

`submission.py` exposes the competition interface: `get_ttt_model(submission_dir, device)`
returns an object with `reset_ttt_state()` and `ttt_step(input_norm, prev_target_norm)`.
The weights are not published in this repository. With a compatible `model.pth`
next to `submission.py`, zip the two files at the archive root (no wrapping
folder) to package it.

## Task

The evaluator streams each trajectory as consecutive 20-frame windows on a
$32 \times 64$ grid. At step $k$ the model receives the context
$x_k \in \mathbb{R}^{20 \times 32 \times 64 \times 2}$ (normalized $u, v$) and must
predict the next 20 frames, $\hat y_k$, optionally with per-element bounds
$[\ell_k, h_k]$. The ground truth $y_{k-1}$ of the *previous* window is revealed at
step $k$, so the model can adapt online but never sees the target it is
currently predicting. Every `ttt_step` call is timed.

The score combines five subscores: relative $L_2$ error, turbulent kinetic
energy (TKE) error, mean-velocity-profile error (MVPE), runtime, and the Safe
Prediction Score (SPS), which rewards narrow intervals that contain the truth.

## Method

The method has two parts: an offline-trained forecaster that maps a context
window to the next 20 frames, and three lightweight corrections applied online
inside `ttt_step` using only the ground truth revealed so far.

### 1. Forecaster

A factorized space–time transformer with $d = 384$, 12 heads and 4 blocks
(9.8M parameters), run in bfloat16.

**Tokens.** Each frame is split into $2 \times 2$ patches, giving $16 \times 32 = 512$
spatial tokens per frame and $20 \times 512$ tokens in total:

$$
z^{(0)}_{t,p} = W_{\text{in}}\,\mathrm{patch}_p(x_t) + P^{\text{sp}}_p + P^{\text{time}}_t ,
$$

with fixed 2D sin–cos spatial embeddings $P^{\text{sp}}$ and learned temporal
embeddings $P^{\text{time}}$.

**Unconditioned blocks (1–2).** Pre-LayerNorm residual blocks with attention
factorized into a spatial pass (over the 512 patches of each frame) and a
temporal pass (over the 20 frames at each patch):

$$
z \leftarrow z + \mathrm{SA}_{\text{space}}(\mathrm{LN}_1 z),\qquad
z \leftarrow z + \mathrm{SA}_{\text{time}}(\mathrm{LN}_2 z),\qquad
z \leftarrow z + \mathrm{MLP}(\mathrm{LN}_3 z).
$$

**Regime code.** Reynolds number and angle of attack are never given as
inputs. The model infers a regime code from the context itself by pooling all
tokens after block 2:

$$
w = \mathrm{MLP}_r\!\Big(\tfrac{1}{|\mathcal{T}|}\textstyle\sum_{(t,p)} \mathrm{LN}_r\, z_{t,p}\Big) \in \mathbb{R}^{384}.
$$

**Conditioned blocks (3–4).** The same block structure, modulated by $w$
through adaLN-Zero, with $\mathrm{Mod}(z;\gamma,\beta) = \mathrm{LN}(z)\odot(1+\gamma) + \beta$:

$$
(\beta_1,\gamma_1,\alpha_1,\beta_2,\gamma_2,\alpha_2) = W_a\,\mathrm{SiLU}(w),
$$

$$
z \leftarrow z + \alpha_1\,\mathrm{SA}_{\text{space}}(\mathrm{Mod}(z;\gamma_1,\beta_1)),\quad
z \leftarrow z + \alpha_1\,\mathrm{SA}_{\text{time}}(\mathrm{Mod}(z;\gamma_1,\beta_1)),\quad
z \leftarrow z + \alpha_2\,\mathrm{MLP}(\mathrm{Mod}(z;\gamma_2,\beta_2)).
$$

(Both attention passes share $\gamma_1, \beta_1, \alpha_1$.)

**Decoder.** All 20 future frames are predicted in one forward pass, with no
autoregressive rollout. Each future frame $\tau$ and patch $p$ gets a query
$q_{\tau,p} = E_\tau + P^{\text{sp}}_p$, which cross-attends over all context
tokens:

$$
o_{\tau,p} = \mathrm{CA}\big(\mathrm{LN}_o\, q_{\tau,p},\ z\big).
$$

**Outputs.** A linear head maps each $o_{\tau,p}$ to its $2 \times 2$ pixels,
giving a mean $\mu$ and a standard deviation
$\sigma = \max(\mathrm{softplus}(s), 10^{-4})$ for each of $u, v$. A second head
predicts a non-negative per-pixel amplitude map from the mean and spread of the
decoder tokens over the forecast horizon:

$$
a = \mathrm{softplus}\Big(W_{\text{tke}}\big[\,\overline{o}^{\,\tau},\ \sqrt{\mathrm{Var}_\tau(o) + 10^{-6}}\,\big]\Big) \in \mathbb{R}_{\ge 0}^{32\times 64 \times 2}.
$$

### 2. Training

Write $\bar z^{\tau}$ for the average of $z$ over the 20 forecast frames. The
amplitude-corrected mean is

$$
m = \bar\mu^{\tau} + a \odot (\mu - \bar\mu^{\tau}),
$$

and the per-pixel turbulent kinetic energy of a field $y$ is

$$
\mathrm{TKE}(y) = \frac{1}{2 \cdot 20} \sum_{\tau=1}^{20} \Big( (y_{\tau,u} - \bar y_u^{\tau})^2 + (y_{\tau,v} - \bar y_v^{\tau})^2 \Big) .
$$

The training loss is

$$
\mathcal{L} = \frac{\lVert m - y \rVert_2}{\lVert y \rVert_2} +
\lambda_\sigma \left( s + \frac{(y - \mathrm{sg}[m])^2}{2\sigma^2} \right) +
\lambda_q \mathcal{L}_{\mathrm{pinball}} +
\lambda_{\mathrm{reg}} \mathcal{L}_{\mathrm{regime}} +
\lambda_{\mathrm{tke}} \frac{\lVert \mathrm{TKE}(m) - \mathrm{TKE}(y) \rVert_2}{\lVert \mathrm{TKE}(y) \rVert_2},
$$

with $\sigma = \mathrm{softplus}(s)$, where

- $\mathrm{sg}$ is stop-gradient,
- $\mathcal{L}_{\mathrm{pinball}}$ trains the quantile channels,
- $\mathcal{L}_{\mathrm{regime}}$ regresses $(\log \mathrm{Re}, \sin\alpha, \cos\alpha)$ from $w$,
- $\lambda_\sigma = \lambda_q = 0.1$ and $\lambda_{\mathrm{reg}} = 0.05$.

**Data.** Real PIV trajectories only (81 trajectories).
Simulation pretraining was tried and hurt: the sim/real turbulence-intensity
mismatch is channel-inverted, so it cannot be fixed by rescaling. Training
windows are 20 context + 20 target frames at stride 5. Inputs are normalized
with the official per-channel statistics. Validation holds out every 6th file
within each angle-of-attack group, so every regime is present in both train and
validation (holding out a whole regime left the model blind to AoA = 20°).
All stages use AdamW, weight decay 0.01, gradient clipping at 1.0, a cosine
schedule with linear warmup, and an EMA of the weights (decay 0.9999).

Training has three stages.

1. **Teacher.** An 8-block, MLP-ratio-4, 25.2M-parameter model is trained from
   scratch on the loss above, followed by a short fine-tune of its amplitude head
   against the per-channel TKE term (the archived teacher checkpoint is at epoch 30
   of that fine-tune). Mirror augmentation $(u, v, \alpha) \mapsto (u, -v, -\alpha)$
   (flip the cross-stream axis, negate $v$) is used when training the backbone.
2. **Distillation.** The 4-block student (MLP ratio 2, 9.8M parameters) is
   initialized from the teacher, not from scratch:
   - Blocks 0–1 (unconditioned) are copied from the teacher's blocks 0–1.
     Blocks 2–3 (conditioned) are copied from the teacher's blocks 2 and 4.
     Keeping two unconditioned blocks keeps the regime pooling in the same place.
   - Each MLP is pruned from 1536 to 768 hidden units. A unit $j$ is scored by
     $\lVert W_1[j,:] \rVert \cdot \lVert W_2[:,j] \rVert$ and the top 768 are kept.
   - The student trains on the same loss with an extra term
     $\lambda_{\mathrm{kd}} \, \lVert m_{\mathrm{s}} - m_{\mathrm{t}} \rVert_2 / \lVert m_{\mathrm{t}} \rVert_2$
     against the frozen teacher's mean prediction ($\lambda_{\mathrm{kd}} = 1$,
     $\lambda_{\mathrm{tke}} = 0$), computed on the same batch each step. The ground-truth
     terms stay on, so the student can correct teacher mistakes.
   - First cycle: 20 epochs, batch 32, peak learning rate 1e-4, 300 warmup steps,
     no mirror augmentation. The 8-to-4 cut kept improving, so training continued
     as 7 warm-restarted cosine cycles (132 epochs in total), each restarting the
     optimizer and schedule. Restarting matters: resuming without it continues at
     the decayed end-of-cycle rate.
3. **Amplitude head.** The student's backbone is frozen and only $W_{\mathrm{tke}}$
   (about 6k parameters) is fine-tuned for 20 epochs with $\lambda_{\mathrm{tke}} = 1$
   and every other loss weight set to 0 (cosine schedule, peak learning rate 2e-4,
   batch 32). This gives the shipped v14 weights.

The teacher's epoch count and learning rate are not given here: the run logs
were deleted and the settings were not recorded elsewhere.

At inference, the quantile channels and the regime-regression heads are not
used.

### 3. Test-time adaptation (`ttt_step`)

The corrections below use only quantities revealed so far in the stream.
Per-trajectory state is cleared by `reset_ttt_state()`; the interval
calibration in step (c) deliberately persists across trajectories.

**(a) Shrunk Welford bias.** The forecaster's error has a persistent spatial
offset within a trajectory. The model tracks the running mean residual of its
raw prediction $\mu$ over the $n$ windows revealed so far, collapses it over
lead time, and adds it back with shrinkage toward zero while $n$ is small:

$$
r_n = y_{k-1} - \mu_{k-1}, \qquad
b_n = b_{n-1} + \frac{r_n - b_{n-1}}{n}, \qquad
\tilde\mu_k = \mu_k + \frac{n}{n + 10}\,\overline{b_n}^{\,\tau}.
$$

**(b) TKE amplitude correction.** The raw forecast is temporally too smooth:
it underestimates fluctuation energy. The amplitude map rescales fluctuations
around the forecast's own temporal mean:

$$
m_k = \overline{\tilde\mu_k}^{\,\tau} + a \odot \big(\tilde\mu_k - \overline{\tilde\mu_k}^{\,\tau}\big).
$$

The bias $\overline{b_n}^{\,\tau}$ is constant over lead time, so it shifts
$\overline{\tilde\mu_k}^{\,\tau}$ and leaves the fluctuations
$\tilde\mu_k - \overline{\tilde\mu_k}^{\,\tau}$ unchanged. The amplitude map
therefore gives the same output whether it is applied before or after the bias
correction, and the head sees the same inputs it was trained on.

**(c) Online interval calibration.** Bounds are $m_k \pm K_{c,\tau}\,\sigma_k$, with
a separate multiplier for each channel $c \in \{u, v\}$ and lead time
$\tau \in \{1,\dots,20\}$. Each multiplier is picked from a fixed grid
$\mathcal{K} = \{0.3, \dots, 10\}$ (20 values) to maximize an empirical version
of the SPS reward, accumulated over revealed windows in physical units:

$$
S_{c,\tau}(K) \mathrel{+}= \frac{1}{|\Omega|}\sum_{i \in \Omega}
\mathbb{1}\big[\,|y_i - m_i| \le K\sigma_i\,\big]\,
\exp\!\Big(-\frac{2K\sigma_i}{\sigma_g}\Big),
\qquad
K_{c,\tau} = \arg\max_{K\in\mathcal{K}} S_{c,\tau}(K),
$$

where $\Omega$ is the set of measured (non-zero) pixels, subsampled by 4, and
$\sigma_g = 0.0564$ is the competition's global scale. This mirrors the SPS
element, which pays $\exp(-\text{width}/\sigma_g)$ only when the truth is
inside the interval. Defaults before any scoring are $K_u = 1.5$ and
$K_v = 2.0$. Scores are updated every step for the first 50 windows and every
5th step after that, to save time.

### Where the corrections help

Measured on 15 held-out real trajectories, using a checkpoint of the same
recipe trained without those trajectories. Gains are in final-score points,
estimated from a linear fit of final_score to the leaderboard subscores (the
official weighting is not published).

| correction | low Re (3,750–5,025) | mid Re (11,400–12,675) | high Re (20,325–22,875) |
|---|---|---|---|
| (a) Welford bias | +1.19 | +0.12 | +0.13 |
| (b) TKE amplitude | +0.07 | +0.27 | +0.49 |

The bias correction matters most in the near-periodic, low-Re regime. The
amplitude correction grows with Re, where the forecast's under-dispersion is
largest.

## License

MIT, see [LICENSE](LICENSE).
