# Streak / No-Streak Binary Classifier — Training Design

**Target:** Per-shot binary classifier on the LCLS attosecond CookieBox 16-eTOF
angular-streaking spectrometer that decides "streak present" vs "unstreaked"
directly from the **raw `Ximg`** (pre-denoising) 2D image (angle × energy).
Output: a single logit. Deployment target: **FPGA inference** (informs
architecture, parameter budget, and calibration choices).

Assumed physics parameters:
- 16 azimuthal detectors (22.5 deg spacing).
- Streak magnitude of interest: ~2 eV kinetic-energy shift, peak-to-peak.
- Energy bin width: 0.25 eV → a 2 eV streak spans ~8 bins.
- Raw noisy input, not the autoencoder's `Ypdf_denoised`.

---

## 1. Physics primer

### 1.1 The CookieBox observable

The LCLS attosecond CookieBox is an **attosecond angular-streaking instrument
consisting of 16 electron time-of-flight (ToF) spectrometers** arranged
azimuthally around the interaction point [1]. X-ray FEL pulses ionize
core-level electrons out of neon: *"The X-ray pulses promote electrons from
the neon core level into an ionization continuum, where they are dressed with
the electric field of a circularly polarized infrared laser"* [2, 3]. The
resulting dressed photoelectrons carry a drift momentum p_drift(t) = p_0 -
e·A_IR(t_ion), so the ionization time is mapped to a direction in the plane
via the rotating IR vector potential A_IR — the "attoclock." Because A_IR is
circular, the KE shift observed at azimuthal angle φ has the form
ΔE(φ) ∝ |A_IR|·√(2 E_0/m)·cos(φ − φ_streak), a **sinusoid across the 16
detectors** whose amplitude encodes the IR-field strength at the moment of
ionization and whose phase encodes the ionization time [2, 4].

### 1.2 What a "streak" is, operationally

A streak is an **angle-dependent modulation of the photoelectron kinetic
energy across azimuthal detectors** [5, 6]. In the (angle, KE) image, a
single-pulse streak looks like a peak that is *shifted up in energy on one
side of the ring and down on the opposite side*, tracing a discrete
half-period of a cosine. From these modulations the single-shot attosecond
intensity structure and chirp of arbitrary SASE X-ray pulses can be
recovered [4]. A **null** ("no-streak") shot has no coherent azimuthal
sinusoid — the per-detector peak positions are dominated by static
detector offsets, shot-to-shot BW jitter, and count noise, with no phase
correlation across φ.

### 1.3 The sinogram / Radon connection

The 16-detector-vs-energy image is a **coarse sinogram**. This is not a
metaphor: the Radon transform of an off-center point source is a sinusoid,
which is precisely why Radon data is called a sinogram [7]. Multiple point
sources yield a superposition of blurred sinusoids with different amplitudes
and phases [8]; the Hough transform is mathematically equivalent to the
Radon transform for parametric curve detection [9]. **Detecting a streak is
therefore an instance of the classical "find a sinusoid in a sinogram"
problem** — a well-studied class in CT/PET reconstruction, and the strongest
piece of non-CV prior art for this classifier.

### 1.4 Matched filtering — the theoretical benchmark

For a **known signal template in additive stochastic noise, the matched
filter is the optimal linear filter for maximizing SNR** [10]. Its impulse
response is a time-reversed conjugate copy of the template: matched
filtering is equivalent to correlating the template with the observation
[11]. For streak detection the "template" is the family of streak images
parameterised by (amplitude, phase φ_streak, KE_peak, X-ray BW), and the
matched-filter output is the coefficient of the best-fitting cosine
across the 16 detectors at each energy row. **Any binary streak/no-streak
classifier we train is upper-bounded in linear-SNR terms by the matched
filter**; a deep model can only beat it by exploiting non-linear structure
(non-Gaussian noise, energy-shape priors, multiplicative gain, coupling
between the streak sinusoid and the SASE bandwidth).

### 1.5 Why the physics helps the classifier

Two properties make streak detection tractable even at low SNR:
1. **Rank-1 structure.** A pure streak is separable: an energy profile
   multiplied by a cosine in φ. The signal lives in a 1-D subspace of the
   16-D angular space at each energy bin.
2. **Circular coherence.** Shot-to-shot fluctuations are uncorrelated across
   detectors (in the ideal case); a streak forces correlation with a specific
   φ-dependence. This is exactly the structure matched filtering and Radon-
   domain feature extraction exploit.

---

## 2. Signal model

### 2.1 Streak footprint

Model the noiseless per-shot image X_signal(φ_i, E) as
```
X_signal(φ_i, E) = A · P(E − ΔE(φ_i))
ΔE(φ_i) = (ΔE_max / 2) · cos(φ_i − φ_streak)
```
with 16 detectors φ_i = i · 22.5°, i = 0..15, ΔE_max ≈ 2 eV peak-to-peak,
and P(E) the intrinsic (streak-free) photoline shape (SASE-broadened,
detector-broadened). At the peak KE row, the streak shifts the line up by
+1 eV on one side and down by −1 eV on the opposite side.

### 2.2 Concrete bin picture

At 0.25 eV/bin:
- Peak-to-peak displacement: 2 eV / 0.25 eV = **8 bins**.
- Sinusoid across φ: 16 samples per full period → **8 samples per half-cycle**
  → well above Nyquist for a single-period signal (16 samples for one full
  spatial period is 8× oversampled relative to Nyquist).

The characteristic feature is not a single-pixel change; it is a
**correlated 8-bin displacement pattern that walks smoothly around the
16-channel ring**. That coherent shape is what the classifier learns; per-
detector noise is uncorrelated with φ and cannot fake it.

### 2.3 SNR bounds

Let σ be the per-(φ, E)-bin noise standard deviation of the raw `Ximg`. The
single-bin SNR of a streak is (ΔE_max/2) · |∂P/∂E| / σ, typically much less
than 1 for weak streaks. The **matched-filter SNR gain** for a template
with N_eff independent bins is √N_eff [10]:
- Angular integration: √16 = 4× SNR gain if the streak template's phase
  φ_streak is known.
- Energy-support integration over the ~8-bin displacement region: another
  √8 ≈ 2.8× gain.
- Combined ceiling: **~11× SNR improvement** over a single-bin decision.
- If φ_streak is unknown a matched-filter bank costs a small log-factor
  (best of ~16 phases) but is otherwise unchanged.

This sets the target: a well-designed classifier should approach this
~11× bound; the exact number depends on the noise covariance across φ (if
detectors share correlated gain drifts, effective N_eff drops).

### 2.4 What distinguishes streak from noise

- **Rank-1 angular structure at fixed energy row.** SVD of the (φ, E) image
  restricted to the photoline energy support: a streak concentrates ~all
  variance in σ_1 with u_1 ≈ cos(φ − φ_streak); noise spreads it across all
  16 σ_i evenly. This is exactly why the existing 2D-SVD denoising stage
  works, and why raw-Ximg classification is still tractable — the sinusoidal
  pattern *is* the leading SVD mode.
- **Phase coherence across energy.** A single streak has one φ_streak; the
  same φ shows up at every energy row inside the photoline. Random noise
  has no φ persistence across energy.

---

## 3. Recommended training-data split

**Recommendation: 50/50 (balanced), with oversampling of the streak class
if the simulator's natural streak rate is below 50%.**

### 3.1 The specific reasoning

The class-imbalance literature is not univocal, but for **weak-signal
detection with a physics-defined threshold** (streak vs no-streak by
ΔE_max), three empirical results point the same way:

1. **Buda, Maki & Mazurowski (2018)** — the largest systematic study of
   class imbalance in CNNs across MNIST/CIFAR-10/ImageNet — concludes:
   *"the method of addressing class imbalance that emerged as dominant in
   almost all analyzed scenarios was oversampling"* [12]. The paper compared
   oversampling, undersampling, two-phase training, and thresholding.
2. **Johnson & Khoshgoftaar (2019)** survey confirms that ROS/RUS and
   cost-sensitive learning transfer to deep learning as strong baselines
   [13].
3. **Focal loss (Lin et al. 2017)** frames the problem as: *"the vast number
   of easy negatives"* overwhelm CE training, and *"focal loss focuses
   training on a sparse set of hard examples"* by down-weighting the loss
   on well-classified examples [14, 15]. This is complementary to balancing
   — it addresses the *within-class* easy/hard split even after balancing.
4. **Class-balanced loss (Cui et al. 2019)** proposes reweighting by
   1/E_n where E_n = (1−β^n)/(1−β) rather than raw inverse frequency, and
   reports significant gains over unweighted training on long-tailed
   CIFAR/ImageNet/iNaturalist [16, 17]. This is a fallback if strict
   oversampling is impractical.

### 3.2 Why not 25/75 or the true prior

- The classifier is a **detector**, not a Bayes-optimal posterior estimator
  on the deployment distribution. Detection theory (Neyman-Pearson) trains
  a discriminant, then sets the operating threshold from the deployment
  prior. Training class ratio should be chosen to make the *discriminant*
  learnable, not to match the prior.
- At weak SNR, negatives (no-streak) form the "easy" class — with a large
  no-streak surplus, cross-entropy gradients are dominated by trivially
  correct negatives, exactly the failure mode focal loss was invented to
  address [14].
- The empirical CNN literature says oversampling is dominant across
  scenarios [12], and 50/50 is the natural target because streak vs no-
  streak is not intrinsically long-tailed — both classes are physically
  well-defined and the "rarity" of one is a data-generation choice, not a
  natural distribution.
- Buda et al. do NOT unambiguously endorse "fully to 50/50 over partial"
  in every case (that stronger claim did not survive verification); the
  defensible recommendation is **train balanced, deploy with threshold
  moving** (§5).

### 3.3 Concrete procedure

1. Simulate (or select) an equal number N of streak and no-streak shots for
   training. If simulator streak rate is p ≠ 0.5, oversample the minority
   with random augmentation (§7) rather than just repeating shots.
2. Sweep streak-magnitude distribution across [0, 4] eV during training
   (see §7). Include a controlled fraction of *near-threshold* streaks
   (0.5–1.5 eV) — these are the hard positives that define the detection
   boundary.
3. Reserve a **validation set drawn at the deployment prior** (not
   balanced) so calibration and threshold selection use realistic base
   rates.
4. If retraining with a strict cost-sensitive framing is preferred, use
   Cui et al.'s class-balanced weighting [16] with β = 0.9999 as a
   drop-in alternative.

---

## 4. Model architecture

Ranking by suitability given the FPGA-inference constraint and the
physics of the observable.

### 4.1 Recommendation: small CNN with matched-filter-friendly first layer

- **Input:** (1, 16, N_E) where N_E is the number of energy bins (~200–800
  depending on window).
- **Layer 1: Conv2D with kernel (16, K_E)** where K_E ≈ 12 bins (covers the
  8-bin streak footprint with 2-bin margin), padding wrapped in φ (angular
  circular padding), 4–8 output channels. **This layer is a learned
  matched-filter bank** across angular phase and energy shape — it is the
  correct inductive bias for a sinogram sinusoid.
- **Angular circular padding** is essential: the streak is periodic in φ.
  Implement by pre-concatenating the last few φ rows to the front, first
  few to the back, before a "valid" convolution — this is cheap on FPGA.
- **Layer 2–3:** Depthwise-separable Conv2D with pooling in the energy axis
  only (angular axis is already collapsed by the (16, K_E) kernel).
- **Head:** Global-average-pool over the energy axis, then a small dense
  layer to a single logit.

Target parameter count: **~3k–4k parameters**, in line with the deployed
CookieBox FCNN (3,433 parameters, chosen over a 3,665-parameter CNN for
better latency-to-resource tradeoff [1]).

### 4.2 Why CNN over pure MLP

- The physics has explicit translation symmetry in energy (streak can occur
  at any photoline center within the SASE BW) and rotational symmetry in
  φ. A CNN with circular φ-padding encodes both; an MLP has to learn them
  from data at the cost of parameters.
- The hls4ml team has demonstrated **5 μs FPGA inference for small CNNs**
  [18], well within the microsecond-scale trigger regime relevant to
  online CookieBox operation.
- hls4ml also demonstrates **sub-microsecond latency for small NN
  classifiers** more broadly [19], so a CNN with a compressed first-layer
  matched-filter bank is compatible with real-time triggering.

### 4.3 Why not pure matched filter

A hand-coded matched-filter bank across (φ_streak, ΔE_max, KE_center) is
the theoretical linear-SNR benchmark [10, 11] and should be implemented
as a **baseline** — but three things make it suboptimal on real data:
1. The photoline shape P(E) drifts with SASE BW and detector transmission
   — a learned first layer adapts.
2. Additive noise is not white or stationary across detectors — the
   whitened matched filter needs the noise covariance, which drifts.
3. Non-linear artifacts (detector saturation, sub-pixel binning) violate
   the linearity assumption.
Ship the matched filter as an interpretable baseline, ship the CNN as the
production classifier.

### 4.4 The CookieBox FCNN precedent

The deployed LCLS-II CookieBox ML uses a **fully connected network with
3,433 parameters, chosen over a CNN with 3,665 parameters for its better
latency-to-resource tradeoff** [1]. That precedent is important: if
compile-time latency in hls4ml pushes the CNN above the deployment
budget, an FCNN of ~3k parameters is the fallback, and the tradeoff study
in [1] is directly transferable to this classifier since it targets the
same detector and platform.

---

## 5. Loss / threshold / calibration

### 5.1 Loss

- **Default: binary cross-entropy (BCE-with-logits)** on the 50/50 training
  set. With balanced classes and small models, plain BCE is well-behaved.
- **If plain BCE is dominated by easy negatives**: switch to focal loss
  with γ = 2, α tuned so positive-class loss weight is 0.25–0.5. Focal
  loss *"down-weights the loss assigned to well-classified examples"* and
  prevents the *"vast number of easy negatives from overwhelming the
  detector during training"* [14]. Practical order of operations: train
  BCE first, look at the loss-per-example histogram; if the CE loss
  saturates on near-zero-loss negatives while positive-class loss is
  still high, switch to focal.
- **Class-balanced loss** [16] with β = 0.9999 is a principled fallback if
  the training set cannot be re-sampled at data-loading time.

### 5.2 Threshold moving at inference

Train balanced (§3), then **shift the operating threshold** at inference
to match the deployment prior. The Bayes-optimal threshold shift for a
prior p_pos (deployment) versus 0.5 (training) is:
```
logit_shift = log(p_pos / (1 − p_pos)) − log(0.5/0.5)
            = log(p_pos / (1 − p_pos))
```
Report ROC and PR curves and let the physicist pick the operating point
by fixed FPR (§6) rather than fixed threshold.

### 5.3 Calibration

Post-hoc **temperature scaling** [20]: fit a single scalar T on a held-out
calibration set to rescale logits before sigmoid. Guo et al.: *"temperature
scaling is surprisingly effective at calibrating predictions"* and often
outperforms more complex methods [20]. Do this after training, on a
deployment-prior validation split, and freeze T for inference. Temperature
scaling is monotone in logit and therefore does not change ROC or PR
metrics — it only changes probability calibration and the numerical
threshold for a given TPR/FPR target.

---

## 6. Augmentation (physics-grounded)

Every augmentation below is a symmetry the deployed detector actually has;
none of them introduce off-manifold synthetic streaks.

1. **Angular rotation.** Roll the 16-channel axis by k ∈ {0..15}. The streak
   phase φ_streak is uniform on [0, 2π) at the source, so this is a free
   augmentation.
2. **Angular reflection.** Flip channel axis. Reflection maps a
   circularly-polarised streak to the opposite helicity — only include if
   the target physics is helicity-agnostic; otherwise reserve as a hold-out
   symmetry.
3. **Energy shift.** Roll energy axis by ±k bins with wrap or edge padding
   (bounded so the photoline stays in-window). Simulates BW jitter,
   nominal-KE drift.
4. **Streak-magnitude scaling.** During training-time positive-shot
   generation, scale ΔE_max ∈ [0.5, 4] eV. **Include ΔE_max = 0 shots
   labeled negative** — this defines the physics-relevant decision
   boundary.
5. **Detector-gain jitter.** Multiply each of the 16 channels by a random
   gain ∼ log-normal(0, σ_g); σ_g calibrated from real per-detector
   fluctuation.
6. **Additive Gaussian noise floor.** Add σ_n · N(0, 1) to each pixel;
   σ_n covers the deployment noise budget.
7. **Detector dropout.** With small probability zero out one full channel
   — simulates a broken eTOF. Forces the classifier to be robust to the
   redundancy the sinogram already provides.

Do **not** augment via generic image transforms (CutMix, MixUp of two
shots, random affine warping in (φ, E)) — these violate the streak-image
manifold and will bias the classifier away from the matched-filter
optimum.

---

## 7. Evaluation

### 7.1 Primary metrics

1. **ROC-AUC** on a validation set drawn at deployment prior.
2. **PR-AUC** — more informative than ROC-AUC if deployment prior is
   heavily biased toward no-streak.
3. **TPR at fixed FPR** — the operationally meaningful number. Report at
   FPR ∈ {1e-2, 1e-3, 1e-4} depending on downstream tolerance for false
   streak triggers.
4. **Sensitivity vs streak magnitude.** TPR as a function of injected
   ΔE_max on synthetic positives, holding FPR fixed on a held-out
   negative pool. This is the plot a physicist reviewer will ask for
   first; it directly reports the classifier's SNR-to-decision curve
   and can be compared against the matched-filter benchmark.

### 7.2 Calibration metric

**Expected Calibration Error (ECE)** on the deployment-prior validation
set, before and after temperature scaling [20].

### 7.3 Matched-filter baseline

Report the same TPR-vs-ΔE_max curve for the analytic matched-filter
detector (§4.3, §1.4). The gap between the classifier curve and the
matched-filter curve quantifies (a) how much non-linear structure the
classifier is exploiting and (b) how close the classifier is to the linear
SNR ceiling [10]. If the classifier is below the matched filter at any
ΔE_max, there is a training or architecture bug.

### 7.4 Suggested figures

- **F1** — Example raw `Ximg` streak vs no-streak side by side, with the
  fitted cos(φ − φ_streak) overlay on the streak.
- **F2** — Loss and ROC-AUC vs epoch, balanced train / prior-shift val.
- **F3** — ROC and PR on the prior-shift validation set, with matched-
  filter baseline overlaid.
- **F4** — TPR-vs-ΔE_max at fixed FPR = 1e-3, classifier vs matched
  filter.
- **F5** — Reliability diagram (probability vs empirical frequency)
  before and after temperature scaling.
- **F6** — Confusion matrix at the chosen operating point on the prior-
  shift validation set.

---

## 8. Open questions

1. **What is the streak-magnitude distribution in the simulator?** The
   decision-boundary is defined by ΔE_max; without a physical minimum
   ΔE_max the training positives may be trivially separable and the
   deployment TPR will collapse on genuine weak streaks. Need a histogram
   of simulated ΔE_max, and a physics-defined "detectable" threshold.
2. **What is the per-detector noise floor and inter-detector covariance
   in the raw `Ximg`?** The matched-filter benchmark needs the noise
   covariance to be tight; if detectors share correlated gain drifts, the
   effective N_eff for the √N_eff SNR gain in §2.3 is smaller than 16.
3. **Is φ_streak distributed uniformly on [0, 2π) or coupled to the
   X-ray arrival time via the IR-carrier phase?** Uniform is a safe
   default and matches the augmentation strategy in §6; a coupled
   distribution changes the balanced-training strategy because "streak"
   and "no-streak" then live at different (φ_streak, KE_center) modes
   and the classifier can memorize the mode rather than learn the
   sinusoidal structure.
4. **Multi-pulse vs single-pulse positives.** The existing `how_many`
   classifier separates pulse counts; is "streak/no-streak" defined on
   *any* number of pulses (n ≥ 1), or only on n = 1? The signal model in
   §2 is single-pulse; multi-pulse superposition changes the rank-1
   angular structure argument (multiple sinusoids of different phase)
   and needs an explicit design decision before training-set generation.

---

## Sources

- [1] Rahimifar et al., "EdgeAI on the LCLS-II CookieBox," Mach. Learn.: Sci.
  Technol. 5, 045041 (2024), https://inspirehep.net/files/d79e656f9fc971afcd18782fb8005153
- [2] LMU Hartmann dissertation, https://epub.ub.uni-muenchen.de/67152/
- [3] Hartmann, Rauschenberger et al., Nat. Photonics 12, 215 (2018),
  https://www.nature.com/articles/s41566-018-0107-6
- [4] LMU Hartmann dissertation, ibid.
- [5] Hartmann et al. 2018, Nat. Photonics, ibid.
- [6] Hartmann et al. 2018, ibid.
- [7] Wikipedia, "Radon transform," https://en.wikipedia.org/wiki/Radon_transform
- [8] Wikipedia, Radon transform, ibid.
- [9] Wikipedia, "Hough transform," https://en.wikipedia.org/wiki/Hough_transform
  (Deans 1981, IEEE PAMI 3(2), 185–188 establishes the equivalence.)
- [10] Wikipedia, "Matched filter," https://en.wikipedia.org/wiki/Matched_filter
  (North 1943; standard result in Van Trees, Kay, Proakis.)
- [11] Matched filter, ibid.
- [12] Buda, Maki, Mazurowski, "A systematic study of the class imbalance
  problem in convolutional neural networks," Neural Networks 106:249–259
  (2018), arXiv:1710.05381, https://arxiv.org/abs/1710.05381
- [13] Johnson & Khoshgoftaar, "Survey on Deep Learning with Class
  Imbalance," J. Big Data 6:27 (2019).
- [14] Lin, Goyal, Girshick, He, Dollár, "Focal Loss for Dense Object
  Detection," ICCV 2017, arXiv:1708.02002, https://arxiv.org/abs/1708.02002
- [15] Focal loss, ibid.
- [16] Cui, Jia, Lin, Song, Belongie, "Class-Balanced Loss Based on Effective
  Number of Samples," CVPR 2019, arXiv:1901.05555,
  https://arxiv.org/abs/1901.05555
- [17] Class-balanced loss, ibid.
- [18] Aarrestad et al., "Fast convolutional neural networks on FPGAs with
  hls4ml" (2021), arXiv:2101.05108, https://arxiv.org/abs/2101.05108
- [19] Duarte et al., "Fast inference of deep neural networks in FPGAs for
  particle physics," JINST 13 P07027 (2018), arXiv:1804.06913,
  https://arxiv.org/abs/1804.06913
- [20] Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural
  Networks," ICML 2017, arXiv:1706.04599, https://arxiv.org/abs/1706.04599
