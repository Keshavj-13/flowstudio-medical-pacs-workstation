#!/usr/bin/env python3
"""Summarize fixed local audits; keep images, weights and per-case records out of Git."""
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path

AUDIT = Path('/var/tmp/cancer-model-eval')
OUT = Path('/workspace/reports/cancer-model-audit-20261111')
ROOT = Path(__file__).resolve().parents[1]

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def mean(rows, key):
    return statistics.mean(row.get(key) or 0.0 for row in rows) if rows else None

def export_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seg = read_rows(AUDIT/'segmentation-rows.jsonl')
    assert len(seg) == 100 and all(row['status'] == 'ok' for row in seg)
    assert len({(r['model'], r['dataset'], r['case_id']) for r in seg}) == 100
    summary = []
    for model in ['cice_beatnet', 'legacy_beatnet']:
        rows = [row for row in seg if row['model'] == model]
        summary.append(dict(model=model, scope='full_volume', n=len(rows), mean_dice=mean(rows,'dice'),
                            median_dice=statistics.median(r['dice'] for r in rows), mean_iou=mean(rows,'iou'),
                            foreground_precision=mean(rows,'precision'), foreground_recall=mean(rows,'recall'),
                            empty_predictions=sum(r['pred_voxels']==0 for r in rows),
                            median_pipeline_seconds=statistics.median(r['pipeline_seconds'] for r in rows)))
    groups = collections.defaultdict(list)
    for row in seg:
        groups[row['dataset'], row['model']].append(row)
    organs = [{'dataset':dataset,'model':model,'n':len(rows),'mean_dice':mean(rows,'dice'),
               'mean_iou':mean(rows,'iou'),'foreground_recall':mean(rows,'recall')}
              for (dataset,model),rows in sorted(groups.items())]
    vlm = []
    all_vlm = []
    for adapter, expected in [('grpo_4040',30),('cot_sft_3d',30),('mp_sft_3d',30),('cesm_2d_sft',20),('base',30)]:
        rows = read_rows(AUDIT/f'vlm-{adapter}-rows.jsonl')
        assert len(rows) == expected and len({r['id'] for r in rows}) == expected
        all_vlm.extend(rows)
        pos = [r for r in rows if r['label']=='positive']
        neg = [r for r in rows if r['label']=='negative']
        valid = sum(r['valid_schema'] for r in rows)
        vlm.append(dict(adapter=adapter,n=len(rows),positive_n=len(pos),negative_n=len(neg),valid_outputs=valid,
                        positive_mean_dice=mean(pos,'dice') if valid else None,
                        positive_mean_iou=mean(pos,'iou') if valid else None,
                        slice_presence_correct=sum(r['presence_correct'] for r in rows),
                        slice_presence_accuracy=mean(rows,'presence_correct') if valid else None,
                        positive_slice_sensitivity=mean(pos,'presence_correct') if valid else None,
                        negative_slice_specificity=mean(neg,'presence_correct') if valid else None,
                        invalid_or_error=sum(r['status']!='ok' for r in rows),
                        median_pipeline_seconds=statistics.median(r['pipeline_seconds'] for r in rows)))
    restored = json.loads((AUDIT/'restoration-summary.json').read_text())
    availability = restored['demo_availability']
    assert not restored['errors'] and all(x['image_exists'] and x['mask_exists'] for x in availability)
    files = [x for dataset in restored['restored'] for x in dataset['files']]
    sources = [ROOT/'genuine_engine.py', ROOT/'legacy_beatnet.py', ROOT/'tools/benchmark_segmentation.py',
               ROOT/'tools/restore_demo_data.py',Path('/workspace/repos/medical-reasoning-studio/backend/reasoning_engine.py'),
               Path('/workspace/repos/medical-reasoning-studio/tools/benchmark_vlm.py')]
    provenance = json.loads((AUDIT/'weight-provenance.json').read_text())
    provenance.update(date_utc='2026-11-11',gpu='NVIDIA H100 MIG; about 39.4 GiB visible VRAM',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        restored_demo_cases=len(availability), downloaded_files=len(files),downloaded_bytes=sum(x['bytes'] for x in files),
        data_root='/var/tmp/data_reservoir',raw_results=str(AUDIT),segmentation_thresholds={'cice_beatnet':0.35,'legacy_beatnet':0.5},
        segmentation_sampling='Fixed validation candidates reconstructed from current loader seeds; 44 candidates plus six existing breast demos. Checkpoint holdout membership not proven.',
        vlm_sampling='15 fixed MRI patients, one reference-positive and one reference-negative axial slice each; 20 CESM saved-validation images, 10 annotated lesions and 10 metadata-normal images.',
        surface_metric='max of two directional 95th-percentile physical surface distances; null when either mask is empty; not the pooled MedPy HD95 convention.',
        prediction_reference_policy='References never supplied to VLM; segmentation prediction called with gt_mask=None. MRI slice sampling uses references, so this is a conditional 2D audit.',
        empty_mask_policy='Dice=1 only if both masks empty; invalid VLM outputs score zero in raw rows and are counted as failures, not silently dropped.',
        timing='Warm inference excluding model initialization and NIfTI loading; includes current pipeline preprocessing. VLM includes image processing, generation and MedSAM2 when invoked. Concurrent server workload may affect latency.',
        gpu_memory_note='Segmentation peak_allocated_gib includes any other model already loaded in the benchmark process; it is not an isolated model memory comparison.')
    (OUT/'protocol.json').write_text(json.dumps(provenance,indent=2))
    for name,rows in [('segmentation_summary',summary),('segmentation_by_dataset',organs),('vlm_summary',vlm),('segmentation_per_case',seg),('vlm_per_slice',all_vlm)]:
        export_csv(OUT/(name+'.csv'),rows)
    pct=lambda x:'unavailable' if x is None else f'{100*x:.2f}%'
    lines=['# Cancer website model audit — 11 November 2026','',
      f'Both websites are running: PACS on port 8070 and reasoning studio on port 8000. All {len(availability)} catalog cases have images and reference masks available; all 22 live PNG slice checks passed.',
      f'Restored {len(files)} selected files ({sum(x["bytes"] for x in files)/2**30:.2f} GiB) under `/var/tmp/data_reservoir`, outside `/workspace`. This is a demo/evaluation subset, not full copies of every training dataset.',
      '','## Fresh full-volume segmentation measurements','',
      'Dice measures mask overlap, not cancer diagnosis accuracy. These scores measure the deployed preprocessing and inference pipeline, not the theoretical ceiling of each network. All 50 volumes per model completed; all-zero or poor predictions remain in the denominator.','',
      '| Model | Volumes | Mean Dice | Median Dice | Mean IoU | Foreground recall | Empty predictions | Median seconds |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["model"]} | {r["n"]} | {pct(r["mean_dice"])} | {pct(r["median_dice"])} | {pct(r["mean_iou"])} | {pct(r["foreground_recall"])} | {r["empty_predictions"]} | {r["median_pipeline_seconds"]:.2f} |')
    lines += ['','| Dataset | Volumes | CICE mean Dice | Legacy mean Dice |','|---|---:|---:|---:|']
    for dataset in sorted({r['dataset'] for r in organs}):
        a=next(r for r in organs if r['dataset']==dataset and r['model']=='cice_beatnet')
        b=next(r for r in organs if r['dataset']==dataset and r['model']=='legacy_beatnet')
        lines.append(f'| {dataset} | {a["n"]} | {pct(a["mean_dice"])} | {pct(b["mean_dice"])} |')
    lines += ['','The sampled BraTS and prostate results are useful demonstration candidates. Breast, abdominal and lung automatic localization need substantial work before claiming broad performance. Except for brain sliding-window inference, the deployed code predicts only one 96×96×32 patch and fills the remaining volume with zero. The legacy checkpoint was trained for breast inputs; its cross-organ scores are an out-of-domain deployment stress test, not evidence of its original breast-validation performance.',
      '4D MRI uses the first sequence as the primary input and the second as the secondary input, matching the live route. MSD brain contains four sequences; the current two-channel path cannot reproduce all four training channels. BraTS uses T1CE plus FLAIR. This limits comparisons across the brain datasets.',
      '','## Fresh VLM + MedSAM2 slice measurements','',
      'MRI results use 30 slices from 15 patients (15 lesion-positive, 15 lesion-negative). A negative slice from a patient with a lesion elsewhere is not a healthy patient. References select the audit slices but do not enter model inference. Training overlap is unknown, so these are local audit scores rather than certified held-out estimates. Positive-slice Dice excludes the easy empty/empty cases.','',
      '| Policy | Inputs | Valid outputs | Positive mean Dice | Presence correct | Positive-slice sensitivity | Negative-slice specificity | Median seconds |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in vlm:
        lines.append(f'| {r["adapter"]} | {r["n"]} | {r["valid_outputs"]}/{r["n"]} | {pct(r["positive_mean_dice"])} | {r["slice_presence_correct"]}/{r["n"]} | {pct(r["positive_slice_sensitivity"])} | {pct(r["negative_slice_specificity"])} | {r["median_pipeline_seconds"]:.2f} |')
    lines += ['','The multi-phase adapter is evaluated exactly as currently wired: one image, not five phases. Its number does not measure temporal kinetic reasoning. CESM uses 20 saved-validation images with 10 annotated lesions and 10 metadata-confirmed normals; an abnormal image lacking a mask was excluded as unknown. CESM and the base model produced no usable complete structured outputs within the configured generation budgets; a diagnostic accuracy cannot be claimed from those runs. CESM training code uses text/token supervision without image tensors, so visual grounding also needs validation.',
      'MedSAM2 contributes to the joint VLM-to-mask scores above; it has not been independently evaluated here with controlled prompts. No fresh BI-RADS accuracy, malignancy accuracy, AUROC, calibration, or patient-level sensitivity/specificity was measured. SAM3 weights and the SP-Reas adapter are missing, so they have no fresh scores.',
      '','## Historical scores and checkpoint provenance','',
      'The deployed CICE checkpoint stores validation mean Dice 79.12% and median Dice 86.81% at epoch 86. A separate historical JSON reports mean 81.73% and median 96.65% across 1,190 samples. Those historical entries use different sample mixes and reference-guided crop evaluation; the JSON has no checkpoint hash. Neither number should be shown as current whole-volume clinical accuracy. Full model and adapter SHA256 hashes and package versions are recorded in protocol.json.',
      'The legacy website loader previously skipped 38 missing and 40 unexpected keys, including attention and decoder weights. It now uses the original SCSE attention decoder, instance normalization and deconvolution settings with strict loading: zero missing/unexpected keys. The CICE loader allows only its six unused deep-supervision head entries and rejects other mismatches.',
      '','## Corrections and validation','',
      'Removed reference-mask-guided inference crops; aligned liver/pancreas organ IDs with the training loader; restored the exact legacy architecture; separated reference-window scores from full-volume Dice; removed unverified historical percentages from the case display; corrected physical voxel volumes; and scored annotated empty masks correctly.',
      'The VLM loader now unloads the previous PEFT adapter when switching policies or selecting Base, avoiding in-place adapter contamination. CESM uses its trained y/x coordinate convention, while the MRI policies use x/y. Streaming has a 120-second token wait timeout. Five PACS integrity checks and three VLM checks passed. Both services are supervised; all 22 live case images rendered and a BraTS GPU scan completed with an HTTP 200 results page.',
      '','## Files and reproduction','',
      '`segmentation_summary.csv`, `segmentation_by_dataset.csv` and `vlm_summary.csv` contain aggregates. `segmentation_per_case.csv` and `vlm_per_slice.csv` retain every case and failure. `protocol.json` records model hashes, source hashes and metric definitions. Raw logs, restoration manifests, pinned dataset revisions and generated audit images are under `/var/tmp/cancer-model-eval`. Earlier invalid-loader/label runs are preserved separately and excluded from the final tables.',
      'Run the restored-code scripts with `/workspace/envs/cancer_web/bin/python`: `tools/restore_demo_data.py`, `tools/benchmark_segmentation.py`, and the VLM repository’s `tools/benchmark_vlm.py --prepare` followed by one `--adapter NAME` process per policy. Existing JSONL results are resumable; use an empty audit directory or archive old rows before intentionally changing the protocol. Run `tools/summarize_model_audit.py` after all evaluations complete. `/var/tmp` is container-local and may disappear if this server/container is replaced.',
      '','## Model storage recommendation','',
      'Use private Hugging Face model repositories for checkpoints/adapters/configs and GitHub for application code. This is worthwhile for these large artifacts: versioned model commits allow exact downloads, rollback and clearer links from deployment to evaluation. No weights were uploaded in this task. A private Hub account/token is needed to publish them.',
      'Hugging Face documents model repositories as Git-based: https://huggingface.co/docs/hub/models-uploading . Dataset sources: https://medicaldecathlon.com/ ; https://huggingface.co/datasets/wanglab/CT_DeepLesion-MedSAM2 ; BraTS mirror https://huggingface.co/datasets/rocky93/BraTS_segmentation . Preserve original dataset licensing and citations.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    # Only aggregates and protocols without patient-level records enter application repositories.
    for repo in [ROOT,Path('/workspace/repos/medical-reasoning-studio')]:
        docs=repo/'docs';docs.mkdir(exist_ok=True)
        (docs/'MODEL_AUDIT_20261111.md').write_text('\n'.join(lines))
    print(json.dumps({'report':str(OUT/'report.md'),'segmentation':summary,'vlm':vlm,'downloaded_files':len(files),'downloaded_gib':sum(x['bytes'] for x in files)/2**30},indent=2))

if __name__=='__main__':
    main()
