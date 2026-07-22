# Claude Code prompt — implement 4 anti–modality-collapse strategies in MultiModalSCVI

*Copy everything below the line into Claude Code, run from the repo root (`zhsegal/PxlgnProject`). It is written so Claude Code first verifies the model structure itself, then implements four independently-toggleable, backward-compatible options, each with a regression test and a shared evaluation.*

---

## MISSION

You are extending a multimodal variational autoencoder (`MultiModalSCVI`, scvi-style) that fuses two single-cell modalities from PixelGen PNA data:

- **abundance** — protein counts, ~159 markers, passed as the main `layer` (e.g. `arcsinh` or `clr`).
- **spatial** — marker-pair colocalization z-scores, ~500 features, passed as an `obsm` modality (e.g. `spatial_asinh5_top500var`).

**The problem:** the joint latent behaves like an abundance-only latent. The spatial modality is effectively ignored (modality collapse), so spatially-defined cell states that abundance misses (e.g. a trogocytosis⁺/CD20⁺CD37⁺HLA-DR⁺ CD8 subset) do not appear in the joint embedding.

You will implement **four strategies** to fix this, each behind its own config flag/enum value, each defaulting to **current behavior** so nothing breaks. Do them as **four separate, reviewable commits**, in order, with a checkpoint after each.

---

## PHASE 0 — Orient yourself and report back BEFORE writing any code

Read and build an accurate mental model of the codebase. Do **not** skip this; the specs below reference exact functions and you must confirm they match the current code.

1. Read these files in full:
   - `multimodalvae.py` — the inner module `MultiModalVAE` (classes `Encoder`, `Decoder`, methods `inference`, `generative`, `loss`, `_weighted_sum`, `get_global_weights`, `get_per_cell_weights`).
   - `multimodalvi.py` — the scvi wrapper `MultiModalSCVI` (`setup_anndata`, `get_latent_representation`, `get_weights`, `get_normalized_expression`).
   - `enums.py` — `D`, `AggMethod`.
   - `metrics.py` and `scvi_utils.py` — existing benchmark/metric utilities (Moran's I, PCA, scib).
   - The notebooks `New_Data/multimodalvi_model.ipynb`, `New_Data/sweep_benchmark.ipynb`, `New_Data/spatial_pca_benchmark.ipynb` — to learn how models are trained and evaluated (loss curves, UMAPs, PPC, Moran's I, scib with `bio=cell_type_annot`, `batch=cell_system`).

2. Confirm (and quote the relevant lines back to me) that:
   - In `MultiModalVAE.inference`, the joint shared posterior is formed in two branches: `experts_method=='POE'` (precision-weighted: `lambda_m = weights[m] * 1/var_m`, then `shared_qzv = 1/Lambda`, `shared_qzm = shared_qzv*eta`) and `experts_method=='MOE'` (`shared_qzm = _weighted_sum(shared_means, weights)`, `shared_qzv = _weighted_sum(shared_vars, weights**2)`).
   - Modality weights come from `get_global_weights()` = `softmax(self.learned_weights)` (for `AOE_GLOBAL_WEIGHTS`), fixed (`AOE_FIXED_WEIGHTS`), or per-cell via `get_per_cell_weights()` = `softmax(weight_encoder(concat_inputs))`.
   - The shared/private scaffolding already exists: encoders output `n_shared + n_private[i]` dims; `inference` splits `shared_*` vs `priv_*`; `modality_z[i] = [shared_z, priv_zs[i]]`; `generative` decodes `modality_z[i]` **only** from its own code (i.e. there is currently **no cross-modal reconstruction**).
   - `loss` contains `unimodal_kl`, `joint_kl`, `free_bits`, `_loss_weights`, and the `spatial_masked` handling that zeroes the spatial KL + likelihood for masked cells, plus the `modality_num==1` missing-spatial masking inside the POE branch (`input==0 all` or `input==1000 all`).
   - Modality index 0 = main/abundance, index 1 = spatial. The missing-spatial special-casing assumes index 1 = spatial.

3. **Report back to me** a short summary of the architecture and the exact insertion points you'll use for each strategy. **Wait for any corrections, then proceed.** (If I don't respond, proceed with your best understanding.)

---

## GLOBAL DESIGN CONSTRAINTS (apply to all 4 strategies)

- **Backward compatibility is mandatory.** Every new behavior is gated by a new keyword arg on `MultiModalVAE.__init__` (and threaded through `MultiModalSCVI.__init__` / `setup_anndata` where needed). Defaults must reproduce **bit-for-bit** the current forward/loss when all new flags are at their defaults. Add a regression test that asserts this.
- **Keep it 2-modality-correct but don't hard-code 2.** The code supports `n_modalities` generally; preserve that. Where the existing code special-cases index 1 (missing-spatial mask), keep that behavior.
- **No silent NaNs.** Clamp variances/precisions as the existing code does (`eps=1e-6`), and add asserts that the loss is finite in tests.
- **Config surface:** prefer explicit kwargs over magic. Document each new kwarg in the `__init__` docstring with its default and a one-line description.
- **Expose new latents.** Anything a downstream notebook needs (e.g. the private blocks in Strategy 4) must be reachable via `MultiModalSCVI.get_latent_representation(..., modality=...)` or a new `representation=` arg.
- After each strategy: run the smoke test + regression test, then commit with a message `feat(fusion): strategy N — <name>`.

---

## STRATEGY 1 — Fix MoE weight collapse

**Diagnosis being fixed:** in the MoE/weighted-sum branch the spatial weight `softmax(learned_weights)[spatial]` can drift to ~0 with no penalty — collapse implemented as a single scalar.

**Implement two complementary, independently-toggleable mechanisms:**

1. **Weight floor.** New kwarg `weight_floor: float = 0.0`. When `> 0`, transform the modality weights so each modality keeps at least `weight_floor` of the mass:
   `w = weight_floor + (1 - n_modalities * weight_floor) * softmax(raw)`.
   Apply inside `get_global_weights` and `get_per_cell_weights` (per-cell: apply per row). Assert `weight_floor * n_modalities < 1`.

2. **Weight-entropy regularizer.** New kwarg `weight_entropy_reg: float = 0.0`. When `> 0`, add to the loss a penalty `weight_entropy_reg * (log(n_modalities) - H(w))`, where `H(w) = -Σ w_m log w_m` is the Shannon entropy of the (post-floor) weight vector. This is `≥ 0`, minimized at uniform weights, and pushes back against collapse. For per-cell weights, average `H` over the minibatch.

**Notes:** these apply to both `AOE_GLOBAL_WEIGHTS`/`AOE_FIXED_WEIGHTS` (global vector) and `AOE_PER_CELL_WEIGHTS`. They are most meaningful for `experts_method=='MOE'` but should also work under `'POE'` (where weights modulate precision). Add the entropy term to `LossOutput` bookkeeping so it shows in loss curves.

**Pros/cons to note in the docstring:** keeps spatial in the average but does not by itself force the shared code to *use* spatial well — pair with cross-reconstruction (Strategy 4 flag) if "new biology" is the goal.

---

## STRATEGY 2 — Calibrate PoE overconfidence

**Diagnosis being fixed:** PoE assumes the two modalities are conditionally independent given z, but the spatial z-scores are a deterministic transform of the same graph as abundance → correlated experts → over-tight joint dominated by the low-variance abundance expert.

**New kwarg** `poe_calibration: Literal[None, 'temperature', 'balanced'] = None`, used only in the `experts_method=='POE'` branch of `inference`.

1. **`'temperature'`** — learn a per-modality precision temperature. Add `self.precision_log_temp = nn.Parameter(torch.zeros(n_modalities))`. Replace `lambda_m = weights[m] * (1/var_m)` with
   `lambda_m = torch.exp(-self.precision_log_temp[m]) * weights[m] * (1/var_m)`.
   Initialized at 1 (so default-off behavior when the param is frozen at 0; but since this branch is only entered when `poe_calibration=='temperature'`, default `None` keeps current code path untouched). This lets the model *learn* to discount an overconfident expert.

2. **`'balanced'`** — normalize each expert's precision to a common scale before the product, so neither modality's *summed* precision can run away. Concretely, per modality compute `lambda_m`, then rescale by its mean over latent dims: `lambda_m = lambda_m / (lambda_m.mean(dim=-1, keepdim=True) + eps)`, then re-apply `weights[m]`. Document that this changes the absolute variance scale of the joint posterior (acceptable — downstream uses the latent mean).

**Preserve** the existing missing-spatial masking (`modality_num==1` zeroing) in both sub-options.

**Pros/cons to note:** heuristic; if it underperforms, Strategy 3 (MVTCAE) achieves the same goal more principledly.

---

## STRATEGY 3 — MVTCAE / total-correlation objective

**Goal:** replace the summed per-modality ELBO with a **total-correlation** objective that explicitly maximizes the mutual information each modality shares with the joint latent, so a modality contributing nothing is penalized. Robust when one modality is high-dim/noisy.

**Reference — read before implementing:** Hwang et al., *Multi-View Representation Learning via Total Correlation Objective* (MVTCAE), NeurIPS 2021, and the reference implementation `gr8joo/MVTCAE` (and the MVTCAE/`mmvae`-style loss in `alawryaguila/multi-view-AE`). **Verify the exact coefficient parameterization against that code — do not trust my recollection of the constants.**

**Structure to implement** (new kwarg `objective: Literal['elbo','mvtcae'] = 'elbo'`, with `tc_alpha: float = 0.5`, `tc_beta: float = 1.0`):

- Keep the existing per-modality encoders and the PoE merge to form the joint posterior `q(z|X)` (reuse Strategy-2-calibrated PoE if set).
- Replace the KL part of `loss` (when `objective=='mvtcae'`) with the MVTCAE decomposition, conceptually:
  - **Reconstruction:** `Σ_m loss_weight_m · E_q[log p(x_m | z)]` (unchanged).
  - **VIB term:** `(1 - tc_alpha) · KL( q(z|X) ‖ p(z) )` — controls total information to the prior.
  - **CVIB / shared term:** `(tc_alpha / n_modalities) · Σ_m KL( q(z|X) ‖ q(z|x_m) )` — pulls the joint posterior toward each unimodal posterior, i.e. **maximizes shared information**; this is the term that punishes ignoring a modality.
  - Scale the combined KL terms by `tc_beta`.
- The unimodal posteriors `q(z|x_m)` are the per-modality encoder Gaussians already computed in `inference` (the `shared_*` slices). Make sure they're passed through to `loss` (extend the `inference` return dict if needed).
- Gate everything: when `objective=='elbo'` (default), run the existing `loss` exactly as now.

**Confirm the precise coefficient form against `gr8joo/MVTCAE` and adjust;** report any discrepancy with the conceptual form above.

**Pros/cons to note:** two new hyperparameters; single shared code, so it does not by itself yield a spatial-specific axis (use Strategy 4 for that).

---

## STRATEGY 4 — MMVAE+ shared/private with cross-modal reconstruction

**Goal:** complete the existing shared/private scaffolding into a proper MMVAE+ model so that a **spatial-private latent** captures structure abundance cannot explain — the embedding of "what abundance misses."

**Reference — read before implementing:** Palumbo et al., *MMVAE+: Disentangling shared and private latent factors in multimodal VAEs*, ICLR 2023, and the MMVAE+ implementation in `alawryaguila/multi-view-AE`. Mirror its mechanics; verify details against that code.

**New kwargs:** `cross_reconstruction: bool = False`, `private_prior: Literal['standard'] = 'standard'` (auxiliary prior on private latents), and reuse the existing `n_private` (set e.g. `n_private=[k_abund, k_spatial]`).

**Implement:**

1. **Separate private posteriors properly.** Encoders already output `n_shared + n_private[i]`. In `inference`, in addition to `priv_means`/`priv_zs`, also keep `priv_vars[i] = vars[..., n_shared:]` and form `q(z_priv_i | x_i) = Normal(priv_means[i], priv_vars[i].sqrt())`. The shared block continues to be PoE/MoE-combined into `w` (the shared code).

2. **Self-reconstruction (as now):** decode modality `i` from `[w, z_priv_i]` where `z_priv_i ~ q(z_priv_i|x_i)`.

3. **Cross-reconstruction (new, the key term):** when `cross_reconstruction=True`, also decode modality `i` from `[w_from_other, z_priv_i_prior]`, where:
   - `w_from_other` is the shared code obtained from the **other** modality's encoder only (e.g. to reconstruct spatial, use the shared posterior `q(w|x_abundance)`),
   - `z_priv_i_prior ~ r(z_priv_i)` is sampled from the **auxiliary prior** (standard Normal), **not** from modality i's encoder.
   This is the MMVAE+ trick: the generating modality's private dims are replaced by prior draws, forcing `w` to be sufficient to reconstruct *both* modalities. Add these cross-recon log-likelihoods to the loss (down-weighted by `_loss_weights`, optionally by a separate `cross_recon_weight: float = 1.0`).

4. **KL terms:** `KL(q(w|·)‖p(w))` for the shared code plus `KL(q(z_priv_i|x_i)‖r(z_priv_i))` for each private block. Keep `free_bits` applicable to private dims so they don't collapse to prior before their decoder learns.

5. **Expose the blocks.** Extend `MultiModalSCVI.get_latent_representation` so a downstream notebook can pull (a) the joint `[w, all private]`, (b) the shared `w` alone, and (c) each modality's private block alone. The spatial-private block is the deliverable for clustering the spatial-defined states.

**Pros/cons to note:** most moving parts (shared/private balance, cross-recon weight, disentanglement); add an eval that checks the spatial-private block isn't just memorizing noise (see metrics below).

---

## EVALUATION — add a shared benchmark + a new "latent usage" metric

Add a script/notebook `New_Data/fusion_strategy_benchmark.ipynb` (mirror `sweep_benchmark.ipynb`) that trains the baseline (current default config) plus one run per strategy on `adata_all_annotated.h5ad` (modalities: abundance `arcsinh` layer + `spatial_asinh5_top500var`; `batch_key='sample'`; annotation `cell_type_annot`) and reports, for each:

1. **Loss curves** (incl. the new entropy/cross-recon terms where present).
2. **Moran's I autocorrelation** of spatial features on the joint latent (reuse `metrics.distr_autocorrelation_in_latent`) — higher = the latent organizes cells along spatial axes.
3. **scib** (`bio=cell_type_annot`, `batch=cell_system`).
4. **PPC** self-reconstruction per modality.
5. **NEW — per-modality latent-usage ablation.** For a trained model, zero the spatial input (set the spatial modality to the model's "missing" sentinel, or all-zeros) and measure: (a) ΔELBO, (b) change in the joint latent (mean cosine distance of per-cell embeddings vs unablated), (c) drop in spatial-feature Moran's I. A model that truly uses spatial should degrade on all three; the current model likely won't. **This is the headline metric** that tells us whether collapse is fixed. Implement it as a reusable function in `metrics.py` (`spatial_latent_usage(model, adata, ...)`).

Print a single summary table: rows = {baseline, S1, S2, S3, S4}, columns = {spatial Moran's I, scib bio, scib batch, ΔELBO-on-spatial-ablation, latent-shift-on-ablation}.

---

## TESTING & ACCEPTANCE CRITERIA

Add `tests/test_fusion_strategies.py` (pytest) covering:

1. **Regression / backward-compat:** with all new flags at defaults, the model's `loss` and `inference` outputs match the pre-change code on a fixed random AnnData + seed (compare to a saved tensor or to a `git stash`-ed baseline). Must be exact.
2. **Smoke test per strategy:** build a tiny synthetic 2-modality AnnData (e.g. 200 cells, 20 abundance + 30 spatial features, 2 batches), and for each of S1–S4 (and a couple of flag combos) run `setup_anndata` → `train(max_epochs=2)` → `get_latent_representation` and assert: loss is finite, latent shape is right, no NaNs/Infs, and for S4 the private-block accessor returns the right dims.
3. **Strategy-1 behavior test:** with `weight_floor=0.2`, assert returned weights never go below ~0.2; with `weight_entropy_reg` large, assert weights stay near-uniform after a few steps.
4. **Strategy-3 sanity:** `objective='mvtcae'` produces a different (lower-collapse) result than `'elbo'` on the synthetic data where the spatial modality carries a planted signal abundance lacks (construct such a synthetic case).

**Definition of done:** all tests pass; the benchmark notebook runs end-to-end on the synthetic data (and is ready to point at the real `.h5ad`); each strategy is a separate commit; the summary table is produced. Then give me a short written report of which strategy moved the latent-usage metric most.

---

## WORKFLOW

1. Phase 0 orientation → report back.
2. Branch `feature/fusion-antitcollapse`.
3. Implement S1 → test → commit. Repeat for S2, S3, S4.
4. Add the benchmark + new metric → commit.
5. Run everything on synthetic data; summarize results and any deviations from this spec (especially the MVTCAE/MMVAE+ coefficient details you verified against the reference repos).

Prefer small, well-commented diffs. When a spec here conflicts with what the actual code requires, follow the code and tell me what you changed and why.
