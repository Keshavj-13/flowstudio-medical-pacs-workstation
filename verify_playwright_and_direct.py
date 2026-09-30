#!/usr/bin/env python3
"""
verify_playwright_and_direct.py
--------------------------------
Comprehensive automated verification script for FlowStudio PACS Multi-Organ Workstation.
Performs:
1. Direct Server HTTP API verification across all cases and models.
2. Playwright headless browser E2E verification of UI rendering, slice scrubbing,
   sequence switching, 3D WebGL solids, and <think> stream disclosure.
3. Compares production Dice (Full-Volume and Lesion ROI) against validation benchmarks.
"""

import os
import sys
import time
import json
import urllib.request
from pathlib import Path

# Create screenshots directory
SCREENSHOTS_DIR = Path("/workspace/cice_pacs_workstation/verification_screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://127.0.0.1:8070"

CASES_TO_TEST = [
    {
        "case_id": "brats_01081",
        "organ": "brain",
        "modality": "3d_mri",
        "val_benchmark": 0.9364,
        "dataset": "BraTS 2021"
    },
    {
        "case_id": "spleen_14",
        "organ": "spleen",
        "modality": "3d_ct",
        "val_benchmark": 0.8905,
        "dataset": "MSD Task09 Spleen"
    },
    {
        "case_id": "yunnan_19",
        "organ": "breast",
        "modality": "3d_mri",
        "val_benchmark": 0.8723,
        "dataset": "Mama-Mia / Yunnan Curated"
    },
    {
        "case_id": "prostate_28",
        "organ": "prostate",
        "modality": "3d_mri",
        "val_benchmark": 0.7934,
        "dataset": "MSD Task05 Prostate"
    },
    {
        "case_id": "colon_001",
        "organ": "colon",
        "modality": "3d_ct",
        "val_benchmark": 0.4491,
        "dataset": "MSD Task10 Colon"
    }
]

MODELS_TO_TEST = [
    "cice_beatnet",
    "legacy_beatnet",
    "qwen_nuclear_vlm",
    "dual_consensus"
]


# ==============================================================================
# PART 1: DIRECT HTTP REQUEST VERIFICATION
# ==============================================================================
def run_direct_api_verification():
    print("=" * 95)
    print("🔬 PART 1: DIRECT HTTP SERVER API VERIFICATION")
    print("=" * 95)

    api_results = []

    for c in CASES_TO_TEST:
        case_id = c["case_id"]
        organ = c["organ"]
        val_bm = c["val_benchmark"]

        print(f"\n📂 Testing Case: {case_id} ({organ.upper()} · {c['dataset']}) | Val Benchmark: {val_bm*100:.1f}%")
        print("-" * 95)

        for model in MODELS_TO_TEST:
            url = f"{BASE_URL}/cases/{case_id}/scan-status?model_type={model}&format=json"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    dt = round((time.time() - t0) * 1000.0, 1)

                    vol_dice = data.get("genuine_3d_dice")
                    roi_dice = data.get("roi_dice")
                    val_exp = data.get("validation_expected_dice", val_bm)
                    pred_vol = data.get("pred_volume_ml")
                    lat = data.get("latency_ms")

                    item = {
                        "case_id": case_id,
                        "organ": organ,
                        "dataset": c["dataset"],
                        "model_id": model,
                        "model_name": data.get("model_name"),
                        "vol_dice": vol_dice,
                        "roi_dice": roi_dice,
                        "validation_benchmark": val_exp,
                        "pred_volume_ml": pred_vol,
                        "latency_ms": lat,
                        "http_ms": dt,
                        "status": "PASS"
                    }
                    api_results.append(item)

                    v_str = f"{vol_dice*100:.2f}%" if vol_dice is not None else "N/A"
                    r_str = f"{roi_dice*100:.2f}%" if roi_dice is not None else "N/A"
                    b_str = f"{val_exp*100:.2f}%"

                    print(f"  ✓ [{model:17}] Vol Dice: {v_str:>7} | ROI Dice: {r_str:>7} | Val Bm: {b_str:>7} | Vol: {pred_vol}mL | Latency: {lat}ms")
            except Exception as e:
                print(f"  ✗ [{model:17}] FAILED: {e}")
                api_results.append({
                    "case_id": case_id,
                    "organ": organ,
                    "dataset": c["dataset"],
                    "model_id": model,
                    "status": "FAIL",
                    "error": str(e)
                })

    return api_results


# ==============================================================================
# PART 2: PLAYWRIGHT E2E BROWSER VERIFICATION
# ==============================================================================
def run_playwright_verification():
    print("\n" + "=" * 95)
    print("🎭 PART 2: PLAYWRIGHT HEADLESS BROWSER E2E VERIFICATION")
    print("=" * 95)

    from playwright.sync_api import sync_playwright
    import base64

    def save_page_screenshot(page_obj, path_str):
        try:
            page_obj.screenshot(path=path_str, timeout=10000, animations="disabled")
        except Exception as e:
            print(f"  [Warning] Screenshot exception on {path_str}: {e}")

    playwright_results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--use-gl=angle",
                "--use-angle=swiftshader"
            ]
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # Test 1: Workstation Homepage & Case Catalog
        print("\n[Playwright Step 1] Navigating to Workstation Home...")
        t0 = time.time()
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_selector(".pacs-header", timeout=10000)
        page.wait_for_selector(".cohort-card", timeout=10000)
        catalog_count = page.locator(".cohort-card").count()
        print(f"  ✓ Workstation loaded in {(time.time()-t0)*1000:.1f}ms. Total cohort cases visible: {catalog_count}")

        home_screenshot = str(SCREENSHOTS_DIR / "01_workstation_catalog.png")
        save_page_screenshot(page, home_screenshot)
        print(f"  ✓ Saved catalog screenshot to: {home_screenshot}")

        # Test 2: Brain mpMRI (BraTS 01081) with CICE-BEATNet
        print("\n[Playwright Step 2] Testing BraTS 01081 with CICE-BEATNet...")
        page.evaluate('() => document.getElementById("start-scan-btn").click()')
        page.wait_for_selector(".scanning-overlay", timeout=10000)
        page.wait_for_selector(".scanning-overlay", state="detached", timeout=45000)
        page.wait_for_timeout(1000)

        # Verify Dossier tabs & 3D canvas
        page.wait_for_selector("#tab-mesh3d-btn", timeout=5000)
        page.wait_for_selector("#tab-think-btn", timeout=5000)
        page.wait_for_selector("#three-tumor-container canvas", timeout=15000)
        print("  ✓ Verified 3D Solid Voxel canvas mounted on Case 1 (BraTS)")

        # Capture Dossier Overview Screenshot
        brats_dossier_shot = str(SCREENSHOTS_DIR / "02_brats_cice_beatnet_dossier.png")
        save_page_screenshot(page, brats_dossier_shot)
        print(f"  ✓ Saved BraTS Dossier screenshot to: {brats_dossier_shot}")

        # Switch to <think> stream tab
        page.locator("#tab-think-btn").first.click(force=True)
        page.wait_for_selector("#think-tab", timeout=10000)
        page.wait_for_timeout(600)
        brats_think_shot = str(SCREENSHOTS_DIR / "03_brats_think_stream.png")
        save_page_screenshot(page, brats_think_shot)
        print(f"  ✓ Saved BraTS <think> stream screenshot to: {brats_think_shot}")

        # Test 3: Spleen 14 with CICE-BEATNet (Sequential Case - Testing 3D Voxel Multi-Case Persistence via SPA Drawer)
        print("\n[Playwright Step 3] Testing Spleen 14 (Sequential case without reload - testing 3D voxel fix)...")
        page.evaluate("htmx.ajax('GET', '/cases/spleen_14', '#main-stage')")
        page.wait_for_selector('#start-scan-btn[hx-post*="spleen_14"]', timeout=10000)
        page.wait_for_timeout(500)
        page.evaluate('() => document.getElementById("start-scan-btn").click()')
        page.wait_for_selector(".scanning-overlay", timeout=10000)
        page.wait_for_selector(".scanning-overlay", state="detached", timeout=45000)
        page.wait_for_timeout(1000)

        # Verify 3D canvas is present on sequential Case 2
        page.wait_for_selector("#three-tumor-container canvas", timeout=15000)
        print("  ✓ Verified 3D Solid Voxel canvas successfully re-mounted on Case 2 (Spleen)!")

        spleen_dossier_shot = str(SCREENSHOTS_DIR / "04_spleen_cice_beatnet_dossier.png")
        save_page_screenshot(page, spleen_dossier_shot)
        print(f"  ✓ Saved Spleen Dossier screenshot to: {spleen_dossier_shot}")

        # Test 4: Yunnan 19 Breast with Qwen3.5 Multimodal VLM
        print("\n[Playwright Step 4] Testing Yunnan 19 with Qwen3.5 Multimodal VLM...")
        page.evaluate("htmx.ajax('GET', '/cases/yunnan_19', '#main-stage')")
        page.wait_for_selector('#start-scan-btn[hx-post*="yunnan_19"]', timeout=10000)
        page.wait_for_timeout(500)
        page.evaluate('() => { document.getElementById("model-select").value = "qwen_nuclear_vlm"; document.getElementById("start-scan-btn").click(); }')
        page.wait_for_selector(".scanning-overlay", timeout=10000)
        page.wait_for_selector(".scanning-overlay", state="detached", timeout=45000)
        page.wait_for_timeout(1000)

        # Switch to <think> stream tab to verify chain-of-thought
        page.locator("#tab-think-btn").first.click(force=True)
        page.wait_for_selector("#think-tab", timeout=10000)
        page.wait_for_timeout(600)

        yunnan_vlm_shot = str(SCREENSHOTS_DIR / "05_yunnan19_vlm_think_stream.png")
        save_page_screenshot(page, yunnan_vlm_shot)
        print(f"  ✓ Saved Yunnan 19 VLM <think> stream screenshot to: {yunnan_vlm_shot}")

        # Test In-Viewer Model Switch to Dual Consensus
        print("\n[Playwright Step 5] Testing in-viewer model switch to Dual-Model Consensus...")
        page.locator("#dossier-model-select").select_option("dual_consensus")
        page.wait_for_selector(".scanning-overlay", timeout=15000)
        page.wait_for_selector(".scanning-overlay", state="detached", timeout=75000)
        page.wait_for_timeout(1000)

        # Verify 3D canvas is present on model switch
        page.wait_for_selector("#three-tumor-container canvas", timeout=15000)
        print("  ✓ Verified 3D Solid Voxel canvas persists after in-viewer model switch!")

        consensus_shot = str(SCREENSHOTS_DIR / "06_yunnan19_dual_consensus.png")
        save_page_screenshot(page, consensus_shot)
        print(f"  ✓ Saved Dual Consensus screenshot to: {consensus_shot}")

        browser.close()
        print("\n✓ Playwright headless suite passed with 100% success!")


# ==============================================================================
# MAIN RUNNER & COMPARISON TABLE
# ==============================================================================
if __name__ == "__main__":
    import sys
    t_start = time.time()
    if "--playwright-only" in sys.argv:
        run_playwright_verification()
        print(f"\n✓ Playwright verification finished in {time.time()-t_start:.1f}s")
        sys.exit(0)

    api_results = run_direct_api_verification()
    run_playwright_verification()

    # Generate Markdown Summary Comparison Table
    print("\n" + "=" * 115)
    print("📊 COMPREHENSIVE VALIDATION VS PRODUCTION BENCHMARK AUDIT")
    print("=" * 115)
    print(f"{'Case ID':<13} | {'Organ':<8} | {'Model ID':<17} | {'Prod Vol Dice':<14} | {'Prod ROI Dice':<14} | {'Val Benchmark':<14} | {'Latency':<8}")
    print("-" * 115)

    for r in api_results:
        if r.get("status") == "PASS":
            v_dice = f"{r['vol_dice']*100:.2f}%" if r.get("vol_dice") is not None else "N/A"
            roi_dice = f"{r['roi_dice']*100:.2f}%" if r.get("roi_dice") is not None else "N/A"
            val_bm = f"{r['validation_benchmark']*100:.2f}%"
            lat = f"{r['latency_ms']:.1f}ms"
            print(f"{r['case_id']:<13} | {r['organ']:<8} | {r['model_id']:<17} | {v_dice:<14} | {roi_dice:<14} | {val_bm:<14} | {lat:<8}")
        else:
            print(f"{r['case_id']:<13} | {r['organ']:<8} | {r['model_id']:<17} | {'ERROR':<14} | {'ERROR':<14} | {'N/A':<14} | {'N/A':<8}")

    print("=" * 115)
    print(f"Total Audit Execution Time: {time.time() - t_start:.2f} seconds")
    print(f"Verification artifacts saved to: {SCREENSHOTS_DIR}")
