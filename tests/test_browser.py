import os
import socket
import threading
import time
from pathlib import Path
import pytest


@pytest.mark.skipif(os.environ.get("RUN_BROWSER_TESTS") != "1", reason="Enable browser checks with RUN_BROWSER_TESTS=1")
def test_select_twice_search_upload_and_mobile(workstation):
    from playwright.sync_api import sync_playwright, expect
    import uvicorn
    app, _, case, image_path, _ = workstation
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    service = uvicorn.Server(uvicorn.Config(app.app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=service.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not service.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            pytest.fail("Test HTTP server did not start")
        time.sleep(0.02)
    artifacts = Path("playwright-results")
    artifacts.mkdir(exist_ok=True)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{port}/")
            expect(page.locator("#cohort-search")).to_be_visible()
            page.screenshot(path=str(artifacts / "desktop-library.png"), full_page=True)
            page.locator("#cohort-search").fill("does-not-exist")
            expect(page.locator("#no-study-matches")).to_be_visible()
            page.locator("#cohort-search").fill("brain")
            for _ in range(2):
                page.locator(f'[data-case-id="{case["case_id"]}"]').click()
                expect(page.locator("#dicom-slice-img")).to_be_visible()
                page.locator("#plane-coronal").click()
                expect(page.locator("#slice-slider")).to_have_attribute("max", "9")
                expect(page.locator("#top-slice-pill")).to_contain_text("coronal")
                page.locator("#cohort-open-trigger").click()
            page.screenshot(path=str(artifacts / "desktop-viewer.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.locator("#m-btn-cohorts").click()
            expect(page.locator("#cohort-search")).to_be_visible()
            page.screenshot(path=str(artifacts / "mobile-library.png"), full_page=True)
            page.locator("#nifti-upload").set_input_files(str(image_path))
            page.get_by_role("button", name="Open volume", exact=True).click()
            expect(page.locator("#start-scan-btn")).to_be_disabled()
            expect(page.locator("#dicom-slice-img")).to_be_visible()
            page.screenshot(path=str(artifacts / "mobile-viewer.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            assert errors == []
            browser.close()
    finally:
        service.should_exit = True
        thread.join(timeout=10)
