# Project Completion Plan — Monocular 3D Building Reconstruction

**Status:** Plan only (no code reorganization yet). Approve this plan, then implementation starts.

**Sources analyzed**
- `ML_Project_Draft_Complete.docx` (project goals / methodology / dataset strategy)
- Six Colab cells pasted in chat (setup → data → model → train → 3D → Gradio)

---

## 1. Verdict: Doc vs Code

| Area | Doc says | Code has | Gap |
|------|----------|----------|-----|
| Pipeline | RGB → dual-head CNN → polygon → LoD1 OBJ | Fully sketched end-to-end | Structure only; needs real data + packaging |
| Architecture | Shared encoder + 2 U-Net decoders | `MultiTaskBuildingNet` (ResNet-34) | Matches |
| Losses / metrics | BCE+Dice, masked SmoothL1; IoU/F1/MAE/RMSE | Implemented | Matches |
| BONAI | Footprint / off-nadir only; **not** metric height | `preprocess_bonai(..., height_field="building_height")` | **Critical mismatch** |
| Height labels | Need separate height / DSM−DTM source | Synthetic heights only (default) | **Blocking for paper numbers** |
| Bangladesh | Cross-domain eval (qualitative ± GT) | Missing | Needed for contribution claim |
| Offset branch | Optional | Not implemented | Defer unless time left |
| Demo | Simple UI | Gradio Cell 6 | Keep; make portable |
| Honesty | No fake numbers; no “BONAI has height” | Comments partly honest; training still synthetic | Enforce in README + report |

**Bottom line:** The Colab prototype is a solid architecture/demo scaffold. Completing the *course project as written in the doc* means (1) packaging the code, (2) fixing the BONAI/height data story, (3) training + evaluating on real data, (4) Bangladesh qualitative demo, (5) report with honest claims.

---

## 2. File layout (map 6 cells → modules)

After approval, create this structure under `d:\3d_building`:

```text
3d_building/
├── ML_Project_Draft_Complete.docx      # existing
├── PROJECT_COMPLETION_PLAN.md          # this plan
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml                    # paths, IMG_SIZE, MAX_HEIGHT_M, epochs, weights
├── src/
│   ├── __init__.py
│   ├── paths.py                        # DATA_ROOT / outputs / checkpoints (Colab + local)
│   ├── data/
│   │   ├── dataset.py                  # SatelliteBuildingDataset (Cell 2)
│   │   ├── synthetic.py                # generate_synthetic_* (Cell 2)
│   │   ├── preprocess_bonai.py         # footprint tiling ONLY (fix height assumption)
│   │   └── preprocess_height.py        # NEW: RGB + height / DSM-DTM → masks/heights
│   ├── models/
│   │   └── multitask_net.py            # Cell 3
│   ├── training/
│   │   ├── losses.py                   # Dice + MultiTaskLoss (Cell 4)
│   │   ├── metrics.py                  # compute_metrics (Cell 4)
│   │   └── loop.py                     # run_epoch, train_model (Cell 4)
│   ├── reconstruction/
│   │   ├── predict.py                  # predict_mask_and_height (Cell 5)
│   │   ├── instances.py                # watershed split (Cell 5)
│   │   └── extrude.py                  # mask_and_height_to_3d_mesh (Cell 5)
│   └── demo/
│       └── app.py                      # Gradio (Cell 6)
├── scripts/
│   ├── prepare_synthetic.py
│   ├── prepare_bonai.py
│   ├── prepare_height_data.py
│   ├── train.py
│   └── infer.py
├── notebooks/
│   └── colab_runner.ipynb              # thin cells that import src.* (optional Colab UX)
└── outputs/  checkpoints/  data/       # gitignored
```

**Cell → file mapping**
| Cell | Becomes |
|------|---------|
| 1 Setup | `requirements.txt` + `src/paths.py` + README install notes |
| 2 Data | `src/data/*.py` + prepare scripts |
| 3 Model | `src/models/multitask_net.py` |
| 4 Train | `src/training/*` + `scripts/train.py` |
| 5 Extrude | `src/reconstruction/*` |
| 6 Gradio | `src/demo/app.py` |

Keep Colab-friendly: `paths.py` uses `/content/...` when present, else local `./data`, `./outputs`, `./checkpoints`.

---

## 3. Phased roadmap

### Phase A — Repo scaffold (½–1 day)
**Do:** Create layout above; move logic out of notebook cells; add `requirements.txt`; README with run instructions; delete temp `_doc_extract.txt` if present.
**Accept:** `python -c "from src.models.multitask_net import MultiTaskBuildingNet"` works; synthetic train script starts one epoch.

### Phase B — Fix data story (1–2 days) — **most important**
**Do:**
1. Rewrite BONAI preprocess to export **images + footprint masks only** (no fake `building_height`).
2. Add a height-data pipeline that writes the same `images/masks/heights/` layout from a real height source.
3. Document clearly in README: synthetic = sanity check only; paper numbers = real height set.

**Recommended student-friendly height options (pick one):**
- **Preferred if available:** Any public RGB + building height / nDSM tiles you already have access to (e.g. city open data with building heights, or DSM−DTM aligned to imagery).
- **Practical Colab fallback:** Use a small curated subset of a published height-building dataset you can download legally; if download fails, keep synthetic for pipeline demo but **do not report those metrics as results** (already stated in Cell 2 comments).
- **Two-stage training (matches doc §4.3):** Pretrain footprint on BONAI → freeze/finetune height head on height-labeled set.

**Accept:** At least one real (or clearly labeled synthetic) train/val split on disk; BONAI path never invents heights.

### Phase C — Train + evaluate (2–4 days)
**Do:** Train with checkpointing; plot curves; report IoU/F1 + height MAE/RMSE on **val/test of the height-labeled set**; geographic split if possible.
**Accept:** `checkpoints/best_model.pt` + a short `results.md` table with real numbers; training curves saved.

### Phase D — 3D + demo polish (1 day)
**Do:** Run extrusion on val samples; Gradio with checkpoint load; portable paths; example images (synthetic + 1–2 real crops if available).
**Accept:** Upload image → mask + height viz + downloadable `.obj`.

### Phase E — Bangladesh contribution (1–2 days)
**Do:** Collect a small set of Bangladeshi overhead crops (Google Earth / open sources with license noted). Run inference only.
- Without GT → qualitative figures only (footprint overlay, height heatmaps, 3D screenshots).
- With any GT → quantitative on that subset.
**Accept:** A `results/bangladesh/` folder of figures for the report; text clearly says qualitative vs quantitative.

### Phase F — Report alignment (1 day)
**Do:** Update draft sections with actual method details, dataset names, metrics, limitations.
**Must keep (doc §5.3):**
- Do not claim BONAI provides metric heights.
- Do not claim extrusion/novelty invented here.
- Watershed = post-hoc instance split, not Mask R-CNN.
- Do not report synthetic metrics as main results.

### Optional Phase G — Offset branch
Only if Phases A–F done and time remains. Add a third head for roof→footprint offset using BONAI; keep core LoD1 path unchanged.

---

## 4. Suggested order of work after approval

1. Create package layout + `requirements.txt` from Cell 1.
2. Port Cells 2–6 into `src/` (behavior-preserving).
3. Fix BONAI height assumption + add height preprocess stub.
4. Train synthetic once (smoke test), then switch to real height data when available.
5. Gradio + Bangladesh qualitative pack.
6. Fill report numbers and honesty notes.

---

## 5. Decisions needed from you

Answer these so Phase B is not blocked:

1. **Runtime:** Stay on Google Colab, or develop locally on this Windows machine (GPU?)?
2. **Height data:** Do you already have access to a height-labeled dataset or DSM/DTM? If yes, name/link. If no, OK to proceed with synthetic for pipeline + BONAI for footprint-only pretrain, and mark height results as limited?
3. **Bangladesh images:** Do you have crops ready, or should we plan a small collection step?
4. **Scope:** Core only (A–F), or also optional offset branch (G)?

**Default if you do not answer:** Local + Colab dual paths; synthetic smoke + BONAI footprint pretrain; height head trained on synthetic until real height data arrives; Bangladesh qualitative only; skip offset branch.

---

## 6. Verification checklist (end-to-end)

- [ ] Install from `requirements.txt` succeeds
- [ ] Synthetic dataset builds; one training epoch completes
- [ ] Dummy forward pass shapes `(N,1,H,W)` for mask and height
- [ ] Checkpoint saves/loads
- [ ] `infer` → `.obj` with ≥1 building on a synthetic sample
- [ ] Gradio launches and returns mask, height viz, mesh
- [ ] README states which metrics are synthetic vs real-data
- [ ] Report figures: pipeline diagram, curves, qualitative 3D, Bangladesh examples

---

## 7. Immediate next step (implementation, after you say “go”)

Implement **Phase A + port Cells 2–6 into `src/`** without changing algorithms yet, then apply the BONAI height fix as the first correctness patch.
