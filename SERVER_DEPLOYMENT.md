# Running server release

Source snapshot: server/snapshot-20261111. Deployment branch: deploy/server-20261111.

The running full-body workstation uses /workspace/repos/flowstudio-server-snapshot on port 8070. Original files remain at /workspace/cice_pacs_workstation. Runtime jobs still use the original jobs directory.

The dedicated VLM studio runs separately from /workspace/repos/medical-reasoning-studio on port 8000. Both use /workspace/envs/cancer_web with stable PyTorch 2.6.0+cu124.

Supervisor configuration: /workspace/run/cancer-web/supervisord.conf. Logs and runtime verification results: /workspace/run/cancer-web/. Start both after a container restart with /workspace/bin/start-cancer-web.sh.

```bash
/workspace/envs/cancer_web/bin/python -m supervisor.supervisorctl -c /workspace/run/cancer-web/supervisord.conf status
curl --fail http://127.0.0.1:8070/healthz
curl --fail http://127.0.0.1:8000/healthz
```

The supervisor restarts failed application processes and survives terminal logout. The container itself must remain running.

Live checks: both HTTP health endpoints returned 200; Chromium loaded both interfaces without JavaScript errors; a real CICE GPU scan completed on Yunnan Case 19. The CICE checkpoint returned an empty mask on that case (Dice 0), so model performance remains a separate unresolved issue.

Only 6 of the 22 configured example studies are currently present. Brain and CT examples referencing the previous /var/tmp/data_reservoir are absent; unavailable studies are visibly disabled. No images have been fabricated or replaced with reference masks.

The historical diagnostic template is preserved here. The earlier PR #1 was based on the older GitHub UI; this deployment preserves the newer server UI and applies targeted backend fixes instead.
