# Versioning model weights

Recommendation: use GitHub for both application repositories and private Hugging Face model repositories for selected model releases. This is worthwhile for recovery, exact version pinning, and moving the apps to another GPU server. It does not make inference faster and does not replace the GPU server.

Hugging Face model repositories provide Git history with Xet storage for large binary artifacts. Use model repositories for release versioning: Hub Storage Buckets are mutable and do not provide Git history. [Official repository documentation](https://huggingface.co/docs/hub/repositories).

As checked on 2026-11-11, the documented free private allowance is 100 GB per free account or organization. Private storage above the included paid allowance is metered; check the actual model inventory before choosing a plan. No model inventory or upload has been performed from this session. [Storage limits and current billing terms](https://huggingface.co/docs/hub/storage-limits).

## Proposed layout

Use separate private model repositories for CICE-BEATNet, the legacy BEATNet checkpoint, and the VLM adapters. Keep the VLM base-model identifier and exact base revision alongside each adapter. An adapter alone is not a complete recoverable model.

Each release should include the inference configuration, preprocessing requirements, compatible model-code commit, checkpoint file hashes, and any required tokenizer/processor configuration. Record the full Hugging Face commit SHA in the application's release manifest. A moving main branch or a filename such as latest.pth is not an exact release reference.

Copy selected best checkpoints and final adapters first. Uploading every optimizer state and intermediate training checkpoint increases transfer time and retained history without helping tomorrow's demonstration. Keep the validated local files for the demonstration and add remote versioning separately.

The code repository ignores model weights, scans, local environments, runtime jobs, and environment secrets. Changes to ignore rules do not remove files already tracked in Git.

## Alternative: DVC with your own storage

DVC stores small versioning metadata in Git while the actual model files live in a remote such as S3 or an SSH-accessible server. Prefer it if you already operate suitable storage and want code/model/data changes linked in one Git workflow. It requires remote storage and credentials administration. [DVC documentation](https://doc.dvc.org/start).

For these two applications, a small set of Hugging Face release repositories is the simpler starting point. DVC becomes more attractive if you need the same workflow for large datasets and experiment outputs. This is an engineering recommendation; the exact cost cannot be calculated without the server's checkpoint sizes and retention needs.

## Current status

No Hugging Face repository has been created, no weights have been uploaded, and no existing local checkpoint paths have been changed. Model backup is still outstanding.
