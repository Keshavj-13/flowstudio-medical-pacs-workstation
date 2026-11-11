"""Run against a live PACS server with PACS_BROWSER_URL=http://127.0.0.1:8070."""
import os
import pytest

@pytest.mark.skipif(not os.getenv("PACS_BROWSER_URL"), reason="Requires a live PACS server and Chromium")
@pytest.mark.parametrize("width,height", [(1366,768),(390,844)])
def test_switching_breast_studies_preserves_visible_viewer_and_stops_playback(width,height):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
        page=browser.new_page(viewport={"width":width,"height":height})
        errors=[]
        page.on("pageerror",lambda error:errors.append(str(error)))
        try:
            page.goto(os.environ["PACS_BROWSER_URL"],wait_until="networkidle")
            for case_id in ["ispy2_353318","yunnan_19","yunnan_9"]:
                page.locator("#quick-study-select").select_option(case_id)
                page.wait_for_function("id => currentCaseId===id",arg=case_id)
                page.wait_for_function("() => document.querySelector('#dicom-slice-img').complete && document.querySelector('#dicom-slice-img').naturalWidth>0")
                image=page.locator("#dicom-slice-img").bounding_box()
                assert image["width"]>100 and image["height"]>100
                header=page.locator(".canvas-hud-top").bounding_box()
                cine=page.locator(".floating-cine-bar").bounding_box()
                viewer=page.locator(".viewer-canvas-area").bounding_box()
                assert header["y"]+header["height"]<=cine["y"]+1
                assert cine["y"]+cine["height"]<=viewer["y"]+1
                assert page.evaluate("document.documentElement.scrollWidth")<=width
                assert page.evaluate("!isCinePlaying && cineInterval===null")
                page.locator("#btn-cine-play").click()
                page.wait_for_function("isCinePlaying && cineInterval!==null")
            page.locator("#quick-study-select").select_option("ispy2_353318")
            page.wait_for_function("currentCaseId==='ispy2_353318'")
            page.wait_for_timeout(400)
            assert page.evaluate("!isCinePlaying && cineInterval===null")
            page.evaluate("() => {loadStudyCase('yunnan_19');setTimeout(()=>loadStudyCase('yunnan_2'),20)}")
            page.wait_for_function("document.querySelector('#case-viewer-container')?.dataset.currentCaseId==='yunnan_2'")
            page.wait_for_timeout(400)
            assert page.evaluate("currentCaseId==='yunnan_2'")
            assert not errors
        finally:
            browser.close()
