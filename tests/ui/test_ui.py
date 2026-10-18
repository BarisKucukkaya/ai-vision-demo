"""Browser checks against DEMO_MOCK=1 (optional: skipped when Playwright or its Chromium is missing)."""
import os
import subprocess
import sys
import time
import uuid
from urllib.parse import urlparse

import httpx
import pytest
from PIL import Image

sync_api = pytest.importorskip("playwright.sync_api")

from tests.conftest import ROOT  # noqa: E402
from tests.test_remote import free_port  # noqa: E402


@pytest.fixture(scope="module")
def samples_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("samples")  # never touch the real samples/ folder


@pytest.fixture(scope="module")
def server(samples_dir):
    port = free_port()
    env = dict(os.environ, DEMO_MOCK="1", DEMO_PRELOAD="0", DEMO_COMBOS="", DEMO_REMOTES="", DEMO_SAMPLES_DIR=str(samples_dir))
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"], cwd=ROOT, env=env)
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    yield f"http://127.0.0.1:{port}"
    proc.kill()
    proc.wait()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # Chromium not installed
            pytest.skip(f"Chromium unavailable: {exc}")
        yield b
        b.close()


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / f"street-{uuid.uuid4().hex[:8]}.png"  # unique name: the server keeps uploads in samples/
    Image.new("RGB", (800, 600), (90, 110, 130)).save(path)
    return str(path)


def wait_uploaded(page, photo):
    """Wait until THIS upload is the photo in use (an auto-selected sample must not count)."""
    page.wait_for_function("name => document.querySelector('#photo-name').textContent.includes(name)", arg=os.path.basename(photo))


@pytest.fixture
def page(browser, server):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    pg = ctx.new_page()
    pg.errors, pg.requests = [], []
    pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("request", lambda r: pg.requests.append(r.url))
    yield pg
    ctx.close()


def load(page, server, photo=None):
    page.goto(server)
    page.wait_for_selector(".pill")
    if photo:
        page.set_input_files("#file", photo)
        wait_uploaded(page, photo)


def detect(page):
    page.click("#btn-detect")
    page.wait_for_selector(".card canvas")
    page.wait_for_function("!document.querySelector('#btn-detect').disabled")


def test_loads_clean_and_offline(page, server, photo):
    load(page, server, photo)
    detect(page)
    assert page.errors == []
    hosts = {urlparse(u).hostname for u in page.requests if u.startswith("http")}
    assert hosts <= {"127.0.0.1"}


def test_cards_in_slide_order_with_pytorch(page, server, photo):
    load(page, server, photo)
    detect(page)
    titles = page.locator(".card h3").all_inner_texts()
    assert titles == ["HOG + SVM", "Faster R-CNN", "RetinaNet"]
    assert "OpenCV · same in every framework" in page.locator(".card").first.inner_text()
    assert page.locator(".banner.warn").first.inner_text().startswith("Test model")
    assert "ms" in page.locator(".bigms").first.inner_text()


def test_framework_switch_and_grey_card(page, server, photo):
    load(page, server, photo)
    page.click("[data-fw=tensorflow]")
    detect(page)
    grey = page.locator(".card.grey")
    assert grey.count() == 1
    assert "Not available on this side" in grey.inner_text() and "KerasHub has no Faster R-CNN" in grey.inner_text()
    assert page.locator(".card h3").all_inner_texts() == ["HOG + SVM", "Faster R-CNN", "RetinaNet"]
    page.click("[data-fw=both]")
    detect(page)
    assert page.locator(".card").count() == 5  # hog, frcnn, grey frcnn, retinanet pytorch + tensorflow
    page.click("#chips .chip >> nth=0")  # switch HOG off
    assert page.locator(".card").count() == 4


def test_threshold_is_client_side_only(page, server, photo):
    load(page, server, photo)
    detect(page)
    before = len([u for u in page.requests if "/compare" in u or "/predict" in u])
    page.locator("#threshold").evaluate("e => { e.value = 0.9; e.dispatchEvent(new Event('input', {bubbles: true})); }")
    assert page.locator("#threshold-out").inner_text() == "0.90"
    rows_high = page.locator(".card >> nth=2 >> tbody tr").count()
    page.locator("#threshold").evaluate("e => { e.value = 0.1; e.dispatchEvent(new Event('input', {bubbles: true})); }")
    rows_low = page.locator(".card >> nth=2 >> tbody tr").count()
    assert rows_low > rows_high
    assert len([u for u in page.requests if "/compare" in u or "/predict" in u]) == before
    assert page.locator(".acard svg text", has_text="HOG").count() >= 1


def test_views_and_analysis(page, server, photo):
    load(page, server, photo)
    page.click("[data-fw=both]")
    detect(page)
    titles = page.locator(".acard h3").all_inner_texts()
    assert titles == ["Threshold sweep", "Class coverage", "Size buckets", "NMS effect", "Agreement matrix", "Cost"]
    assert "same algorithm, two frameworks" in page.locator(".acard tr.hl").first.inner_text()
    page.click("[data-view=overlay]")
    assert page.locator(".compare .handle").count() == 1
    page.locator(".compare .handle").focus()
    page.keyboard.press("ArrowRight")
    assert int(page.locator(".compare .handle").get_attribute("aria-valuenow")) == 52
    page.click("[data-view=diff]")
    assert "agreement" in page.locator(".legend").inner_text()
    page.click("[data-view=cards]")
    page.click("text=Compare with NMS off")
    page.wait_for_selector("text=removed")
    assert page.locator("th", has_text="NMS off").count() == 1


def test_table_rows_and_raw_json(page, server, photo):
    load(page, server, photo)
    detect(page)
    rows = {r.locator("th").inner_text(): r.inner_text() for r in page.locator("#table-wrap tbody tr").all()}
    for name in ["Family", "Features", "How it searches", "Classes", "Speed", "Accuracy", "Score meaning", "Settings used"]:
        assert name in rows
    assert "not measured here" in rows["Accuracy"] and "Classic machine learning" in rows["Family"]
    assert "can detect 1 classes" in rows["Classes"] or "can detect 1 class" in rows["Classes"]
    page.locator(".card >> nth=0 >> summary").click()
    text = page.locator(".card >> nth=0 >> pre").inner_text()
    assert '"algorithm": "hog_svm"' in text
    assert "curl -F" in page.locator(".card >> nth=0 >> .curl").inner_text()
    page.click("#btn-theme")
    assert page.evaluate("document.documentElement.dataset.theme") in ("dark", "light")


def test_code_panel_tabs_render_real_code(page, server, photo):
    load(page, server, photo)
    expected = httpx.get(f"{server}/api/code/hog_predict").json()
    page.wait_for_selector(".codebox")
    shown = page.locator(".codebox").first.inner_text()
    assert "detectMultiScale" in shown and str(expected["start_line"]) in shown
    assert page.locator(".codebox .badge").count() >= 3
    tabs = page.locator(".tab").all_inner_texts()
    assert tabs == ["HOG + SVM", "Faster R-CNN", "RetinaNet", "NMS in 15 lines", "Shared post-processing", "Adapter contract", "Framework check", "Training: loop vs fit"]
    for i in range(len(tabs)):
        page.locator(".tab").nth(i).click()
        page.wait_for_function(f"document.querySelectorAll('.codebox').length >= {2 if i >= 6 else 1}")
        assert page.locator(".codebox").first.inner_text().strip()
    page.locator(".tab").nth(6).click()
    page.wait_for_function("document.querySelectorAll('.codebox').length === 2")
    assert page.locator(".panes .pane h4").all_inner_texts() == ["PyTorch (torchvision)", "TensorFlow (KerasHub)"]
    assert page.locator(".notes li").count() == 3
    page.locator(".tab").nth(7).click()
    page.wait_for_selector(".codestat")


def test_fallback_to_recorded_when_server_blocked(page, server, photo):
    page.route("**/health", lambda route: route.abort())
    page.route("**/api/code/**", lambda route: route.abort())
    page.goto(server)
    page.wait_for_selector(".banner.rec")
    banners = page.locator(".banner").all_inner_texts()
    assert any("MOCK DATA" in b for b in banners) and any("Server not reachable" in b for b in banners)
    assert page.locator(".card canvas").count() == 3
    page.wait_for_selector(".codebox")
    assert "(snapshot)" in page.locator("#filebar").inner_text()
    assert page.locator("text=Compare with NMS off").is_disabled()
    assert page.locator("#btn-repeat").is_disabled()
    assert page.locator("#mode-switch [aria-checked=true]").inner_text() == "Recorded"


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (390, 844)])
def test_viewports_have_no_horizontal_scroll(browser, server, photo, size):
    ctx = browser.new_context(viewport={"width": size[0], "height": size[1]})
    pg = ctx.new_page()
    pg.goto(server)
    pg.wait_for_selector(".pill")
    pg.set_input_files("#file", photo)
    wait_uploaded(pg, photo)
    pg.click("[data-fw=both]")
    pg.click("#btn-detect")
    pg.wait_for_selector(".card canvas")
    overflow = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    assert overflow <= 1, f"horizontal overflow {overflow}px at {size}"
    font = pg.evaluate("parseFloat(getComputedStyle(document.body).fontSize)")
    if size[0] == 1920:
        assert font >= 20
    (ROOT / "tests" / "ui" / "screenshots").mkdir(exist_ok=True)
    pg.screenshot(path=str(ROOT / "tests" / "ui" / "screenshots" / f"ui-{size[0]}.png"), full_page=True)
    ctx.close()


def test_keyboard_shortcuts(page, server, photo):
    load(page, server, photo)
    page.keyboard.press("d")
    page.wait_for_selector(".card canvas")
    page.keyboard.press("t")
    assert page.evaluate("document.documentElement.dataset.theme") in ("dark", "light")
    page.locator("#threshold").focus()
    theme = page.evaluate("document.documentElement.dataset.theme")
    page.keyboard.press("t")  # ignored: a slider has focus
    assert page.evaluate("document.documentElement.dataset.theme") == theme


def test_code_panel_collapses_and_remembers(page, server, photo):
    load(page, server, photo)
    page.wait_for_selector(".codebox")
    open_width = page.locator("#code-panel").bounding_box()["width"]
    assert page.locator("#btn-code").get_attribute("aria-expanded") == "true"
    page.click("#btn-code")
    assert page.locator("#btn-code").get_attribute("aria-expanded") == "false"
    assert page.locator("#code-body").is_hidden()
    assert page.locator("#code-panel").bounding_box()["width"] < open_width / 4
    page.reload()
    page.wait_for_selector(".pill")
    assert page.locator("#code-body").is_hidden()  # remembered
    page.keyboard.press("c")
    assert page.locator("#code-body").is_visible() and page.locator("#btn-code").inner_text().endswith("Collapse")


def test_overlay_drag_does_not_select(page, server, photo):
    load(page, server, photo)
    page.click("[data-fw=both]")
    detect(page)
    page.click("[data-view=overlay]")
    page.locator(".compare").scroll_into_view_if_needed()  # mouse events outside the viewport would miss it
    box = page.locator(".compare").bounding_box()
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + box["width"] * 0.2, y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.8, y, steps=12)
    mid = int(page.locator(".compare .handle").get_attribute("aria-valuenow"))
    page.mouse.move(box["x"] + box["width"] * 0.95, y + 200, steps=6)  # even leaving the image keeps dragging
    page.mouse.up()
    assert 70 <= mid <= 90
    assert page.evaluate("getSelection().toString()") == "" and page.evaluate("getSelection().isCollapsed")
    assert int(page.locator(".compare .handle").get_attribute("aria-valuenow")) >= mid


def test_upload_is_used_but_never_stored(page, server, photo, samples_dir):
    page.goto(server)
    page.wait_for_selector(".pill")
    page.set_input_files("#file", photo)
    wait_uploaded(page, photo)
    detect(page)
    assert page.locator("#keep-photo").count() == 0
    assert list(samples_dir.iterdir()) == []  # nothing is written to samples/
    assert page.locator(".thumb").count() == 0
    posts = [u for u in page.requests if u.endswith("/samples")]
    assert posts == [] or all(u == server + "/samples" for u in posts)  # only the GET listing
