# Cancer website model audit — 11 November 2026

Both websites are running: PACS on port 8070 and reasoning studio on port 8000. All 22 catalog cases have images and reference masks available; all 22 live PNG slice checks passed.
Restored 138 selected files (1.81 GiB) under `/var/tmp/data_reservoir`, outside `/workspace`. This is a demo/evaluation subset, not full copies of every training dataset.

## Fresh full-volume segmentation measurements

Dice measures mask overlap, not cancer diagnosis accuracy. These scores measure the deployed preprocessing and inference pipeline, not the theoretical ceiling of each network. All 50 volumes per model completed; all-zero or poor predictions remain in the denominator.

| Model | Volumes | Mean Dice | Median Dice | Mean IoU | Foreground recall | Empty predictions | Median seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| cice_beatnet | 50 | 15.45% | 0.00% | 13.22% | 18.09% | 18 | 0.58 |
| legacy_beatnet | 50 | 0.27% | 0.00% | 0.14% | 0.14% | 0 | 0.45 |

| Dataset | Volumes | CICE mean Dice | Legacy mean Dice |
|---|---:|---:|---:|
| brats2021_full | 5 | 88.00% | 1.51% |
| breast_demo | 6 | 0.00% | 0.00% |
| deeplesion_medsam2 | 5 | 4.36% | 0.00% |
| msd_brain_task01 | 5 | 0.73% | 0.75% |
| msd_colon_task10 | 5 | 0.00% | 0.00% |
| msd_liver_task03 | 5 | 0.00% | 0.00% |
| msd_lung_task06 | 5 | 0.00% | 0.00% |
| msd_pancreas_task07 | 5 | 0.00% | 0.00% |
| msd_prostate_task05 | 4 | 76.73% | 0.51% |
| msd_spleen_task09 | 5 | 0.00% | 0.01% |

The sampled BraTS and prostate results are useful demonstration candidates. Breast, abdominal and lung automatic localization need substantial work before claiming broad performance. Except for brain sliding-window inference, the deployed code predicts only one 96×96×32 patch and fills the remaining volume with zero. The legacy checkpoint was trained for breast inputs; its cross-organ scores are an out-of-domain deployment stress test, not evidence of its original breast-validation performance.
4D MRI uses the first sequence as the primary input and the second as the secondary input, matching the live route. MSD brain contains four sequences; the current two-channel path cannot reproduce all four training channels. BraTS uses T1CE plus FLAIR. This limits comparisons across the brain datasets.

## Fresh VLM + MedSAM2 slice measurements

MRI results use 30 slices from 15 patients (15 lesion-positive, 15 lesion-negative). A negative slice from a patient with a lesion elsewhere is not a healthy patient. References select the audit slices but do not enter model inference. Training overlap is unknown, so these are local audit scores rather than certified held-out estimates. Positive-slice Dice excludes the easy empty/empty cases.

| Policy | Inputs | Valid outputs | Positive mean Dice | Presence correct | Positive-slice sensitivity | Negative-slice specificity | Median seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| grpo_4040 | 30 | 30/30 | 74.37% | 27/30 | 100.00% | 80.00% | 0.89 |
| cot_sft_3d | 30 | 30/30 | 67.60% | 27/30 | 100.00% | 80.00% | 6.63 |
| mp_sft_3d | 30 | 30/30 | 70.38% | 27/30 | 100.00% | 80.00% | 1.32 |
| cesm_2d_sft | 20 | 0/20 | unavailable | 0/20 | unavailable | unavailable | 11.60 |
| base | 30 | 0/30 | unavailable | 0/30 | unavailable | unavailable | 0.67 |

The multi-phase adapter is evaluated exactly as currently wired: one image, not five phases. Its number does not measure temporal kinetic reasoning. CESM uses 20 saved-validation images with 10 annotated lesions and 10 metadata-confirmed normals; an abnormal image lacking a mask was excluded as unknown. CESM and the base model produced no usable complete structured outputs within the configured generation budgets; a diagnostic accuracy cannot be claimed from those runs. CESM training code uses text/token supervision without image tensors, so visual grounding also needs validation.
MedSAM2 contributes to the joint VLM-to-mask scores above; it has not been independently evaluated here with controlled prompts. No fresh BI-RADS accuracy, malignancy accuracy, AUROC, calibration, or patient-level sensitivity/specificity was measured. SAM3 weights and the SP-Reas adapter are missing, so they have no fresh scores.

## Historical scores and checkpoint provenance

The deployed CICE checkpoint stores validation mean Dice 79.12% and median Dice 86.81% at epoch 86. A separate historical JSON reports mean 81.73% and median 96.65% across 1,190 samples. Those historical entries use different sample mixes and reference-guided crop evaluation; the JSON has no checkpoint hash. Neither number should be shown as current whole-volume clinical accuracy. Full model and adapter SHA256 hashes and package versions are recorded in protocol.json.
The legacy website loader previously skipped 38 missing and 40 unexpected keys, including attention and decoder weights. It now uses the original SCSE attention decoder, instance normalization and deconvolution settings with strict loading: zero missing/unexpected keys. The CICE loader allows only its six unused deep-supervision head entries and rejects other mismatches.

## Corrections and validation

Removed reference-mask-guided inference crops; aligned liver/pancreas organ IDs with the training loader; restored the exact legacy architecture; separated reference-window scores from full-volume Dice; removed unverified historical percentages from the case display; corrected physical voxel volumes; and scored annotated empty masks correctly.
The VLM loader now unloads the previous PEFT adapter when switching policies or selecting Base, avoiding in-place adapter contamination. CESM uses its trained y/x coordinate convention, while the MRI policies use x/y. Streaming has a 120-second token wait timeout. Five PACS integrity checks and three VLM checks passed. Both services are supervised; all 22 live case images rendered and a BraTS GPU scan completed with an HTTP 200 results page.

## Files and reproduction

`segmentation_summary.csv`, `segmentation_by_dataset.csv` and `vlm_summary.csv` contain aggregates. `segmentation_per_case.csv` and `vlm_per_slice.csv` retain every case and failure. `protocol.json` records model hashes, source hashes and metric definitions. Raw logs, restoration manifests, pinned dataset revisions and generated audit images are under `/var/tmp/cancer-model-eval`. Earlier invalid-loader/label runs are preserved separately and excluded from the final tables.
Run the restored-code scripts with `/workspace/envs/cancer_web/bin/python`: `tools/restore_demo_data.py`, `tools/benchmark_segmentation.py`, and the VLM repository’s `tools/benchmark_vlm.py --prepare` followed by one `--adapter NAME` process per policy. Existing JSONL results are resumable; use an empty audit directory or archive old rows before intentionally changing the protocol. Run `tools/summarize_model_audit.py` after all evaluations complete. `/var/tmp` is container-local and may disappear if this server/container is replaced.

## Model storage recommendation

Use private Hugging Face model repositories for checkpoints/adapters/configs and GitHub for application code. This is worthwhile for these large artifacts: versioned model commits allow exact downloads, rollback and clearer links from deployment to evaluation. No weights were uploaded in this task. A private Hub account/token is needed to publish them.
Hugging Face documents model repositories as Git-based: https://huggingface.co/docs/hub/models-uploading . Dataset sources: https://medicaldecathlon.com/ ; https://huggingface.co/datasets/wanglab/CT_DeepLesion-MedSAM2 ; BraTS mirror https://huggingface.co/datasets/rocky93/BraTS_segmentation . Preserve original dataset licensing and citations.
