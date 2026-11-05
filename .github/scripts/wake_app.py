"""Keep Streamlit Community Cloud apps awake by visiting them in a real browser.

Usage: wake_app.py [URL ...]   (or set STREAMLIT_APP_URL, comma-separated)

Prints one AWAKE / WOKEN / FAILED line per app. Exits non-zero if any FAILED.
On failure a screenshot is saved to ./screenshots/ (uploaded by the workflow).
"""
import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

SLEEP_TEXT = "This app has gone to sleep due to inactivity"
WAKE_BUTTON = "Yes, get this app back up!"
APP_SELECTOR = '[data-testid="stAppViewContainer"]'
SHOT_DIR = Path(os.environ.get("SCREENSHOT_DIR", "screenshots"))

PAGE_TIMEOUT_MS = 60_000
WAKE_TIMEOUT_S = 180
DWELL_S = 18


def parse_urls(argv):
    raw = argv if argv else [os.environ.get("STREAMLIT_APP_URL", "")]
    urls = []
    for item in raw:
        urls += [u.strip() for u in item.replace("\n", ",").split(",") if u.strip()]
    return urls


def is_asleep(page):
    # Streamlit renders the app inside the main page; the sleep screen is plain text.
    try:
        return page.get_by_text(SLEEP_TEXT).first.is_visible(timeout=1000) or \
            page.get_by_role("button", name=WAKE_BUTTON).first.is_visible(timeout=1000)
    except PlaywrightTimeout:
        return False


def app_loaded(page):
    """The running app lives in an iframe on streamlit.app, so look in every frame."""
    return any(f.locator(APP_SELECTOR).count() > 0 for f in page.frames)


def wait_for_app(page, timeout_s):
    """Poll until the app container exists and the sleep text is gone."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if app_loaded(page) and not is_asleep(page):
            return True
        time.sleep(3)
    return False


def check_app(browser, url):
    """Returns 'AWAKE' or 'WOKEN'; raises on failure."""
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()
    try:
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
            page.wait_for_load_state("networkidle", timeout=15_000)
        except PlaywrightTimeout:
            pass  # Streamlit keeps a websocket open; networkidle may never fire
        time.sleep(5)  # let the sleep screen / app render

        if is_asleep(page):
            page.get_by_role("button", name=WAKE_BUTTON).first.click()
            if not wait_for_app(page, WAKE_TIMEOUT_S):
                raise RuntimeError(f"app did not load within {WAKE_TIMEOUT_S}s after clicking wake button")
            return "WOKEN"

        # Not asleep: confirm the app really rendered, then dwell like a real visitor.
        if not wait_for_app(page, PAGE_TIMEOUT_MS / 1000):
            raise RuntimeError("app container never appeared (and no sleep screen detected)")
        time.sleep(DWELL_S)
        return "AWAKE"
    except Exception:
        SHOT_DIR.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:80]
        try:
            page.screenshot(path=str(SHOT_DIR / f"{slug}.png"), full_page=True)
        except Exception as shot_err:  # never mask the original error
            print(f"  (screenshot failed: {shot_err})", flush=True)
        raise
    finally:
        context.close()


def main():
    urls = parse_urls(sys.argv[1:])
    if not urls:
        print("FAILED no URL given (pass args or set STREAMLIT_APP_URL)")
        return 2

    failed = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for url in urls:
            try:
                status = check_app(browser, url)
                print(f"{status} {url}", flush=True)
            except Exception as e:
                failed += 1
                reason = str(e).splitlines()[0] if str(e) else type(e).__name__
                print(f"FAILED {url} - {type(e).__name__}: {reason}", flush=True)
        browser.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
