Below is a **debug checklist** that maps every high-impact problem I found in the repository to the exact file / function where it lives, explains the downstream symptom you are probably seeing (loss behaviour, GPU OOM, bad clustering, etc.), and gives one-line remedies. I group them roughly from *must-fix* (they break correctness) → *should-fix* (they degrade quality) → *nice-to-fix*.

---

## 1 Show-stoppers – training is mathematically or logistically wrong

| # | File → function                                      | Bug / oversight                                                                                                                                                                                                                                            | What you will observe                                                                                                                     | Quick fix                                                                                                                                                           |
| - | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | **`models/simclr3d.py → SwinTransformer3D`**         | Only a *pseudo-3-D* slice-loop Swin is wired into `SimCLR`; the real 3-D Swin you implemented in `models/swin3d.py` is never used.                                                                                                                         | With `--backbone swin3d` the network still sees 2-D slices, so it fails to exploit 3-D context (silhouette < 0.05 even after 100 epochs). | Replace the import path in `SimCLR` to point to `TrueSwin3D` *or* delete the slice-loop class and re-export the real one.                                           |
| 2 | **`train_simclr.py → import autocast, GradScaler`**  | You import from `torch.amp`, but PyTorch 1.9 (pinned in `requirements.txt`) exposes AMP under `torch.cuda.amp`.                                                                                                                                            | Script crashes at start with “`No module named torch.amp`”.                                                                               | `from torch.cuda.amp import autocast, GradScaler`.                                                                                                                  |
| 3 | **`datasets/bbox_loader.py → process_bbox`**         | You hard-code **three** channels (raw, pre, cleft) and choose “pre” as *always* the first side, ignoring vesicle overlap logic.                                                                                                                            | Model never sees post-synaptic membrane, vesicle cloud or mitochondria; it can’t learn polarity patterns.                                 | Keep all five binary masks **or** at least run the vesicle-overlap rule to decide which side is pre.                                                                |
| 4 | **`train_simclr.py → train_epoch`**                  | Scheduler is stepped every time the **optimizer** steps, but `T_max` was computed as `epochs × (len_loader)` (number of *raw* mini-batches). Because you use gradient-accumulation, the LR decays `gradient_accumulation_steps` times slower than planned. | LR stays near its peak; loss plateaus early, never falls < 4.0.                                                                           | Divide `steps_per_epoch` by `gradient_accumulation_steps` when you create the scheduler, **or** call `scheduler.step()` every raw mini-batch (before accumulation). |
| 5 | **`random_cube.py → Augmenter.apply_gaussian_blur`** | You blur one 2-D slice at a time inside a loop → O(D) Python calls and *massive* Python-GPU sync.                                                                                                                                                          | GPU utilisation < 20 %; training 2× slower than expected.                                                                                 | Move blur to CPU before `.to(device)` or implement a true 3-D separable blur with `F.conv3d`.                                                                       |
| 6 | **`bbox_loader.py → get_closest_component_mask`**    | When the centroid of a vesicle component lies **outside** the crop (because the component touches the volume edge), the distance becomes `nan`, `argmin` chooses 0 and you silently return an empty mask.                                                  | \~15 % of sampled cubes have zero vesicle signal; SimCLR treats them like negatives and collapses contrast.                               | Filter `nan`s before `argmin`; if all distances `nan`, keep *all* vesicle voxels.                                                                                   |

---

## 2 Quality killers – model trains but learns weak features

| #  | Area                       | Issue                                                                                     | Impact                                                               | Mitigation                                                                                                                          |
| -- | -------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 7  | **Channel selection**      | Dropping post-synapse and vesicle masks removed the strongest polarity cue.               | Cluster purity stuck at \~0.10.                                      | Use a 5-channel tensor or merge all masks into one multi-hot channel.                                                               |
| 8  | **Cube centre**            | Sampling is 100 % cleft-centred, 0 voxel jitter.                                          | Encoder can cheat by looking at the centre voxel; NT-Xent collapses. | Add ±8 voxel random shift to `(d_start, h_start, w_start)`.                                                                         |
| 9  | **NT-Xent (loop version)** | Per-row loop builds logits → O(N²) Python operations; slow on big batches.                | Cannot go past batch = 256 on a single GPU without timeout.          | Switch to the vectorised formulation you already wrote in `models/swin3d.py` or reuse lightly modified `torch.nn.CrossEntropyLoss`. |
| 10 | **Gaussian blur kernel**   | Using average-pool as a blur severely smears edges; high-sigma can wipe the entire cleft. | Augmenter sometimes produces blank cubes → noisy gradients.          | Limit σ ≤ 1.0 or implement proper Gaussian.                                                                                         |
| 11 | **Swin memory**            | Keeping every 80³ token map until back-prop uses > 30 GB @ batch = 32.                    | Surreptitious OOM on A100 = 40 GB.                                   | Reduce `patch_size` to (4,4,4) and use gradient checkpointing (`torch.utils.checkpoint` in Swin blocks).                            |

---

## 3 Efficiency / reproducibility nits

| #  | File                                        | Nit                                                                                                                                                                                             | Why care                                                                       |
| -- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 12 | `random_cube.py → _check_component_overlap` | You compare `cleft_volume > threshold * cube_size**3`; that ignores *fraction* of original component—mask dropout or crop-resize can still pass even if only 1 % of the original cleft remains. | Mask-aware guarantee is weak; views drift semantically apart.                  |
| 13 | `nt_xent_loss` (both versions)              | You don’t push the diagonal of the similarity matrix to `-∞` when building full‐matrix losses (vectorised form).                                                                                | Small but measurable influence: own-view logits leak into softmax denominator. |
| 14 | AMP + BN                                    | `SyncBatchNorm` + `autocast` **O1** will overflow half-precision variance when batch << 64.                                                                                                     | Random NaNs after \~30 epochs if you keep physical batch = 32.                 |
| 15 | `requirements.txt`                          | `torch==1.9` cannot import Swin3D from `torchvision 0.19` (released later).                                                                                                                     | Users following README will get `AttributeError: swin3d not found`.            |

---

## 4 Quick sanity checks once you patch the above

| Metric                                   | Healthy range after 50 epochs |
| ---------------------------------------- | ----------------------------- |
| Training loss (2 × batch  = 1024)        | 0.8 – 1.2                     |
| Cosine(z₁,z₂)                            | climbs from 0.05 → 0.25       |
| Silhouette (100 k embeds, k-means = 100) | ≥ 0.15                        |
| GPU util @ batch = 128 phys., acc = 8    | ≥ 70 %                        |

If any of these stay flat, re-check the numbered items in exactly that order—early errors propagate.

---

### One-sentence takeaway

Wire in the **real 3-D Swin**, import AMP correctly, feed **all structural masks**, add cube-centre jitter, blur in 3-D not slice loops, and fix the vesicle-component edge case—these six changes move the pipeline from “runs but stalls” to “converges and clusters synapses meaningfully.”
