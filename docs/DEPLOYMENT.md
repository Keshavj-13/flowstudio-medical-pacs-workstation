# Server deployment handoff

## Current status

The changes in this branch are based on the existing GitHub FlowStudio source, not a copy of the server's uncommitted directories.

Terminal execution in the editing session fails before any shell command starts:

```text
bwrap: Failed to make / slave: Permission denied
```

Consequently, neither app has been started, restarted, installed as a service, or verified on the real GPU server by this change. GitHub Actions checks exercise a temporary test server with artificial test volumes and CPU metric calculations.

## Application locations reported in the original server investigation

| Application | Reported directory | Reported port |
| --- | --- | --- |
| Full-body PACS | /workspace/cice_pacs_workstation | 8070 |
| Dedicated VLM studio | /workspace/reasoning/grpo_reasoning_data/medical_reasoning_studio | 8000 |

These paths and ports come from the supplied investigation and still require verification in a working terminal. The PACS repository is Keshavj-13/flowstudio-medical-pacs-workstation. No repository containing the standalone VLM studio was found in the connected GitHub sources.

## Completion steps when server access works

1. Inspect both directories for AGENTS.md instructions, Git remotes, uncommitted changes, and the actual launchers. Preserve existing work.
2. Back up both application sources to their intended Git remotes, excluding local credentials, scan data, environments, and model weights. Confirm the commit SHA and remote branch for each app.
3. Apply and validate the FlowStudio changes in a clean checkout or an isolated release directory. Do not overwrite a dirty running checkout.
4. Use the established GPU Python environments and confirm dataset mounts and the actual checkpoint files. The viewer-only environment is not an inference environment.
5. Run a real scan with each required model. Confirm output shapes, source affine, physical volume, and the absence of reference-mask substitution.
6. Start the VLM studio with its inspected launcher. Verify its actual checkpoint choice, WebSocket connection, model output, and segmentation handoff.
7. Run the two apps as separate supervised services with restart-on-failure, persistent logs, and the existing ports. Confirm they survive logout and that a failed app does not stop the other.
8. Verify both user-facing URLs from outside the server and retain stable ingress for the demonstration. A URL text artifact from an earlier temporary tunnel is not a health check.

## Quick HTTP checks

From a working server terminal after starting the respective services:

```bash
curl --fail --silent --show-error http://127.0.0.1:8070/healthz
curl --fail --silent --show-error http://127.0.0.1:8070/cases
curl --fail --silent --show-error --output /dev/null http://127.0.0.1:8000/
```

These checks establish HTTP availability only. They do not establish GPU model readiness or VLM generation. Both real inference checks remain necessary.
