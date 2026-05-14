"""Google Scholar scraper using Playwright with a persistent browser profile.

Why a persistent profile: Google Scholar fingerprints headless browsers aggressively
and serves a CAPTCHA within a few searches. A persistent user-data-dir keeps cookies
and localStorage across runs, which dramatically reduces CAPTCHA frequency.

If a CAPTCHA does appear, the script pauses and lets the user solve it manually in the
visible browser window, then resumes.
"""

from __future__ import annotations

import asyncio
import contextvars
import re
import shutil
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from platformdirs import user_data_dir

from . import dbg
from .models import PaperHit

# Module-level shared browser context, set by ``shared_session``. When set, the
# search / fetch_versions / fetch_bibtex_for_cluster functions reuse this one
# context instead of launching a fresh Chromium for each call. This is the
# difference between Scholar treating us as a normal user vs. a bot.
_SHARED_CTX: contextvars.ContextVar[Optional[object]] = contextvars.ContextVar(
    "bibsync_shared_browser_ctx", default=None
)

SCHOLAR_BASE = "https://scholar.google.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Stealth init script — runs on every page BEFORE any site JS, patching the
# automation fingerprints Google Scholar uses to decide whether to CAPTCHA-wall us.
# Compiled from the stealth techniques in playwright-stealth and puppeteer-extra-stealth.
# This is what reliably drops Scholar's CAPTCHA frequency from "every few queries"
# to "very rarely".
_STEALTH_INIT_SCRIPT = r"""
// 1. Hide navigator.webdriver (the single most reliable bot signal).
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

// 2. Spoof navigator.plugins to look like a real Chrome install.
Object.defineProperty(navigator, 'plugins', {
  get: () => [
    { name: 'PDF Viewer', filename: 'internal-pdf-viewer' },
    { name: 'Chrome PDF Viewer', filename: 'internal-pdf-viewer' },
    { name: 'Chromium PDF Viewer', filename: 'internal-pdf-viewer' },
    { name: 'Microsoft Edge PDF Viewer', filename: 'internal-pdf-viewer' },
    { name: 'WebKit built-in PDF', filename: 'internal-pdf-viewer' },
  ],
});

// 3. navigator.languages — real browsers always return >= 2 entries.
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });

// 4. Spoof the chrome runtime global that real Chrome exposes (Playwright omits it).
if (!window.chrome) { window.chrome = {}; }
window.chrome.runtime = window.chrome.runtime || {};

// 5. Permissions API: real Chrome returns the actual permission state for
//    notifications; Playwright's default returns 'denied' unconditionally, which
//    is a tell.
const _origPermQuery = (window.navigator.permissions && window.navigator.permissions.query)
  ? window.navigator.permissions.query.bind(window.navigator.permissions)
  : null;
if (_origPermQuery) {
  window.navigator.permissions.query = (params) =>
    params && params.name === 'notifications'
      ? Promise.resolve({ state: Notification.permission })
      : _origPermQuery(params);
}

// 6. WebGL vendor / renderer — automation frameworks expose a tell-tale "SwiftShader"
//    or generic Mesa string. Spoof to a normal Intel/Apple GPU profile.
try {
  const _origGetParam = WebGLRenderingContext.prototype.getParameter;
  WebGLRenderingContext.prototype.getParameter = function (parameter) {
    if (parameter === 37445) return 'Intel Inc.';
    if (parameter === 37446) return 'Intel Iris OpenGL Engine';
    return _origGetParam.call(this, parameter);
  };
} catch (e) { /* ignore */ }

// 7. Hide the headless-chromium feature flags lingering in navigator.userAgentData.
if (navigator.userAgentData) {
  Object.defineProperty(navigator.userAgentData, 'brands', {
    get: () => [
      { brand: 'Google Chrome', version: '124' },
      { brand: 'Chromium', version: '124' },
      { brand: 'Not-A.Brand', version: '99' },
    ],
  });
}
"""


def _profile_dir() -> Path:
    p = Path(user_data_dir("bibsync", "bibsync")) / "chrome-profile"
    p.mkdir(parents=True, exist_ok=True)
    return p


def reset_profile() -> Path:
    """Delete the persistent Chrome profile. Use when Scholar has flagged the session
    and even solving the CAPTCHA doesn't restore search results."""
    p = _profile_dir()
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    return p


@asynccontextmanager
async def _browser(headless: bool = False) -> AsyncIterator:
    # Lazy import so `bibsync --help` works before `playwright install`.
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(_profile_dir()),
            headless=headless,
            user_agent=USER_AGENT,
            viewport={"width": 1366, "height": 800},  # common laptop resolution
            locale="en-US",
            timezone_id="America/Los_Angeles",
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-site-isolation-trials",
            ],
        )
        # Apply the stealth init script to every page opened in this context.
        # Runs BEFORE any page JS, so Scholar's fingerprint checks see the
        # patched globals from the very first request.
        try:
            await ctx.add_init_script(_STEALTH_INIT_SCRIPT)
            dbg.trace("scholar.browser", "stealth init script applied")
        except Exception as e:
            dbg.trace("scholar.browser", "WARN stealth init failed", error=str(e))
        try:
            yield ctx
        finally:
            await ctx.close()


@asynccontextmanager
async def shared_session(headless: bool = False) -> AsyncIterator:
    """Open one browser context and make it the default for all Scholar calls in this run.

    Without this, every call to :func:`search` / :func:`fetch_versions` /
    :func:`fetch_bibtex_for_cluster` launches a fresh Chromium — one ``fix`` run can
    spawn 12+ browsers, which Scholar reads as a bot. With this, the whole run shares
    one session and looks like a normal user.
    """
    async with _browser(headless=headless) as ctx:
        token = _SHARED_CTX.set(ctx)
        empty_token = _CONSECUTIVE_EMPTY_COUNT.set(0)
        try:
            yield ctx
        finally:
            _SHARED_CTX.reset(token)
            _CONSECUTIVE_EMPTY_COUNT.reset(empty_token)


@asynccontextmanager
async def _ctx_or_new(headless: bool) -> AsyncIterator:
    """Yield the shared context if one is active, otherwise create a one-shot context."""
    existing = _SHARED_CTX.get()
    if existing is not None:
        yield existing
    else:
        async with _browser(headless=headless) as ctx:
            yield ctx


async def _await_no_captcha(page, timeout_ms: int = 120_000) -> None:
    """If a CAPTCHA / 'unusual traffic' page is shown, wait for the user to solve it."""
    url = page.url
    body = (await page.content()).lower()
    captcha_markers = ("unusual traffic", "/sorry/", "recaptcha", "captcha")
    if any(m in url.lower() or m in body for m in captcha_markers):
        print("\n[bibsync] Google Scholar is showing a CAPTCHA.")
        print(
            "[bibsync] Please solve it in the browser window — I'll wait up to 2 minutes."
        )
        # Wait for navigation away from the CAPTCHA page.
        async def _solved() -> bool:
            try:
                u = page.url.lower()
                b = (await page.content()).lower()
                return not any(m in u or m in b for m in captcha_markers)
            except Exception:
                return False

        elapsed = 0
        while elapsed < timeout_ms:
            if await _solved():
                return
            await asyncio.sleep(2)
            elapsed += 2000
        raise RuntimeError("Timed out waiting for CAPTCHA to be solved.")


def _parse_search_results(html: str) -> list[PaperHit]:
    """Parse the result list from a Scholar search page.

    Designed to survive Scholar HTML drift:
      * Class string is matched lenient — any ``<div>`` with the word ``gs_r`` in its
        class list and a ``data-cid`` attribute, in either attribute order.
      * Block boundary is the start of the NEXT result, not a fixed closing pattern.
        This avoids the failure mode where extra nested divs broke the old regex's
        ``</div>\\s*</div>\\s*</div>`` terminator.
    """
    hits: list[PaperHit] = []

    # Find every result-block start position + cluster id, regardless of whether
    # data-cid appears before or after the class attribute.
    starts: list[tuple[int, str]] = []
    for m in re.finditer(
        r'<div\s+[^>]*\bclass="[^"]*\bgs_r\b[^"]*"[^>]*\bdata-cid="([^"]+)"',
        html,
    ):
        starts.append((m.start(), m.group(1)))
    for m in re.finditer(
        r'<div\s+[^>]*\bdata-cid="([^"]+)"[^>]*\bclass="[^"]*\bgs_r\b[^"]*"',
        html,
    ):
        starts.append((m.start(), m.group(1)))

    # De-duplicate by start offset and sort in document order.
    seen_offsets: set[int] = set()
    ordered: list[tuple[int, str]] = []
    for off, cid in sorted(starts):
        if off not in seen_offsets:
            seen_offsets.add(off)
            ordered.append((off, cid))

    for i, (start, cluster_id) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else len(html)
        block = html[start:end]

        title_m = re.search(
            r'<h3\s+[^>]*\bclass="[^"]*\bgs_rt\b[^"]*"[^>]*>(.*?)</h3>',
            block,
            flags=re.DOTALL,
        )
        if not title_m:
            continue
        title_html = title_m.group(1)
        title = re.sub(r"<[^>]+>", "", title_html)
        title = re.sub(r"\s+", " ", title).replace("[PDF]", "").replace("[HTML]", "").strip()
        title = title.lstrip("[").rstrip("]").strip()
        # Strip zero-width directional markers Scholar appends (U+200E, U+200F).
        title = title.replace("‎", "").replace("‏", "").strip()

        authors_block_m = re.search(
            r'<div\s+[^>]*\bclass="[^"]*\bgs_a\b[^"]*"[^>]*>(.*?)</div>',
            block,
            flags=re.DOTALL,
        )
        authors: list[str] = []
        year: Optional[int] = None
        venue: Optional[str] = None
        if authors_block_m:
            ab = re.sub(r"<[^>]+>", "", authors_block_m.group(1))
            ab = re.sub(r"&nbsp;|&#x2026;|…", " ", ab)
            parts = [p.strip() for p in ab.split(" - ")]
            if parts:
                authors = [a.strip() for a in parts[0].split(",") if a.strip()]
            if len(parts) >= 2:
                middle = parts[1]
                year_m = re.search(r"\b(19|20)\d{2}\b", middle)
                if year_m:
                    year = int(year_m.group(0))
                venue_part = re.sub(r",?\s*\b(19|20)\d{2}\b", "", middle).strip().strip(",")
                venue = venue_part or None

        # Citation count: try several patterns since Scholar's HTML varies (and
        # localisation/HTML-entity quirks creep in).
        cited_by = 0
        for pat in (
            r"Cited by[\s&nbsp;]+(\d+)",
            r">Cited by\s*(?:<[^>]+>)?\s*(\d+)",
            r'href="[^"]*\bcites=\d+[^"]*"[^>]*>[^<]*?(\d+)',
            r"Cited by[^0-9]{1,8}(\d+)",
        ):
            cm = re.search(pat, block)
            if cm:
                try:
                    cited_by = int(cm.group(1))
                    break
                except (ValueError, IndexError):
                    continue

        versions_url: Optional[str] = None
        versions_m = re.search(
            r'href="(/scholar\?cluster=[^"]+)"[^>]*>\s*All\s*(\d+)\s*versions', block
        )
        if versions_m:
            versions_url = SCHOLAR_BASE + versions_m.group(1).replace("&amp;", "&")

        snippet_m = re.search(
            r'<div\s+[^>]*\bclass="[^"]*\bgs_rs\b[^"]*"[^>]*>(.*?)</div>',
            block,
            flags=re.DOTALL,
        )
        snippet = None
        if snippet_m:
            snippet = re.sub(r"<[^>]+>", "", snippet_m.group(1)).strip()

        hits.append(
            PaperHit(
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                cited_by=cited_by,
                cluster_id=cluster_id,
                versions_url=versions_url,
                raw_snippet=snippet,
            )
        )

    return hits


def _diagnose_empty_result(html: str, query: str) -> str:
    """When the result parser returns [], inspect the HTML to figure out *why*."""
    low = html.lower()
    if "did not match any articles" in low:
        return f"Scholar reports no matching articles for {query!r}."
    if any(m in low for m in ("captcha", "unusual traffic", "/sorry/", "recaptcha")):
        return (
            "Scholar is showing a CAPTCHA / bot-check page. Solve it in the browser "
            "window and re-run, or run `bibsync config reset-profile` if it persists."
        )
    if "gs_r" not in low:
        return (
            f"Empty result for {query!r}. Scholar returned a page but no result "
            "containers — this is a SOFT BLOCK (your IP/profile is rate-limited). "
            "Fix: run `bibsync config reset-profile` and try again, or wait ~30 min."
        )
    return f"Empty result for {query!r} (parser found result containers but extracted 0 hits — Scholar may have changed its HTML)."


# Module-level counter for consecutive empty results within a single run. Reset by
# shared_session(); incremented by search() on empty results. A run that hits N empties
# in a row almost certainly means a soft block — the caller can surface a clear message.
_CONSECUTIVE_EMPTY_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar(
    "bibsync_consecutive_empty_count", default=0
)


def consecutive_empty_count() -> int:
    """Return how many consecutive search() calls returned empty in this run."""
    return _CONSECUTIVE_EMPTY_COUNT.get()


def _debug_html_path() -> Path:
    """Where we save the last 0-hits page for inspection."""
    return Path(user_data_dir("bibsync", "bibsync")) / "last-empty-search.html"


async def _save_debug_html(html: str, query: str) -> None:
    """Persist the raw HTML of a 0-hit search so the user can inspect what Scholar returned."""
    try:
        p = _debug_html_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        header = f"<!-- query: {query} -->\n"
        p.write_text(header + html, encoding="utf-8")
        print(f"[bibsync] Saved page HTML for inspection: {p}")
    except OSError:
        pass


async def search(query: str, *, headless: bool = False, max_results: int = 10) -> list[PaperHit]:
    """Search Google Scholar by *typing into the search box*, then waiting for results.

    Wait strategy (Scholar loads results in two waves — DOM, then async citation counts):
      1. Wait for the result container (``div.gs_r`` / alert / captcha) — up to 25s.
      2. Wait for citation-count anchors (``a[href*="cites="]``) to populate — up to 12s.
         These load AFTER the initial DOM, and without this wait every parsed hit has
         ``cited_by=0`` (which breaks the cited-by-first sort downstream).
      3. Wait for network idle (no requests for 500ms) — up to 8s.
      4. Final 2.5s jitter so any deferred JS finishes.
    """
    dbg.trace("scholar.search", query=query, max_results=max_results)
    async with _ctx_or_new(headless=headless) as ctx:
        page = await ctx.new_page()
        try:
            dbg.trace("scholar.search", "navigating to home")
            await page.goto(f"{SCHOLAR_BASE}/", wait_until="domcontentloaded")
            await _await_no_captcha(page)
            await asyncio.sleep(0.5)  # let home page settle

            search_box = page.locator("input[name='q']").first
            try:
                await search_box.wait_for(timeout=15_000)
            except Exception:
                dbg.trace("scholar.search", "search box not found; falling back to URL nav")
                await page.goto(
                    f"{SCHOLAR_BASE}/scholar?q={urllib.parse.quote(query)}&hl=en",
                    wait_until="domcontentloaded",
                )
            else:
                dbg.trace("scholar.search", "typing query into search box")
                await search_box.fill("")
                await search_box.type(query, delay=45)
                await asyncio.sleep(0.3)
                await search_box.press("Enter")

            dbg.trace("scholar.search", "waiting for result containers (up to 25s)")
            try:
                await page.wait_for_selector(
                    "div.gs_r, .gs_alrt, form#gs_captcha_f",
                    timeout=25_000,
                )
                dbg.trace("scholar.search", "result containers appeared")
            except Exception:
                dbg.trace("scholar.search", "WARN: result containers timed out")

            await _await_no_captcha(page)

            # Critical: wait for citation-count anchors. Scholar injects these AFTER
            # the initial result DOM; reading too early gives cited_by=0 everywhere.
            dbg.trace("scholar.search", "waiting for citation anchors to populate (up to 12s)")
            try:
                await page.wait_for_function(
                    "() => document.querySelectorAll('a[href*=\"cites=\"]').length > 0",
                    timeout=12_000,
                )
                dbg.trace("scholar.search", "citation anchors present")
            except Exception:
                dbg.trace(
                    "scholar.search",
                    "WARN: citation anchors did not appear in 12s; "
                    "cited_by may be 0 for parsed hits",
                )

            # Let any deferred JS finish (network idle + final jitter).
            try:
                await page.wait_for_load_state("networkidle", timeout=8_000)
                dbg.trace("scholar.search", "network idle")
            except Exception:
                dbg.trace("scholar.search", "networkidle timed out (continuing)")
            await asyncio.sleep(2.5)

            html = await page.content()
            dbg.trace("scholar.search", "got HTML", chars=len(html))
            hits = _parse_search_results(html)[:max_results]
            dbg.trace("scholar.search", f"parsed {len(hits)} hits")
            for i, h in enumerate(hits[:5]):
                dbg.trace(
                    "scholar.parse",
                    f"hit[{i}]",
                    title=h.title,
                    authors=(h.authors[0] if h.authors else ""),
                    year=h.year,
                    cited=h.cited_by,
                    cluster=h.cluster_id,
                )
            if not hits:
                print(f"[bibsync] {_diagnose_empty_result(html, query)}")
                await _save_debug_html(html, query)
                _CONSECUTIVE_EMPTY_COUNT.set(_CONSECUTIVE_EMPTY_COUNT.get() + 1)
                dbg.trace(
                    "scholar.search",
                    "empty result",
                    consecutive=_CONSECUTIVE_EMPTY_COUNT.get(),
                )
            else:
                _CONSECUTIVE_EMPTY_COUNT.set(0)
            return hits
        finally:
            await page.close()


async def fetch_versions(versions_url: str, *, headless: bool = False) -> list[PaperHit]:
    """Fetch the 'All N versions' page for a paper cluster."""
    dbg.trace("scholar.versions", url=versions_url)
    async with _ctx_or_new(headless=headless) as ctx:
        page = await ctx.new_page()
        try:
            await page.goto(versions_url, wait_until="domcontentloaded")
            await _await_no_captcha(page)
            try:
                await page.wait_for_selector(
                    "div.gs_r, .gs_alrt, form#gs_captcha_f", timeout=10_000
                )
            except Exception:
                dbg.trace("scholar.versions", "WARN: wait_for_selector timed out")
            await asyncio.sleep(0.4)
            html = await page.content()
            hits = _parse_search_results(html)
            dbg.trace("scholar.versions", f"parsed {len(hits)} version(s)")
            return hits
        finally:
            await page.close()


async def fetch_bibtex_for_cluster(cluster_id: str, *, headless: bool = False) -> str:
    """Click the Cite icon on a result with this cluster id and download the BibTeX.

    We re-search via cluster URL (one cluster = one result, deterministic) to land on a
    page where the cite icon is present, then drive the modal click flow.
    """
    from playwright.async_api import TimeoutError as PWTimeout

    # Search-by-cluster gives a single result we can click reliably.
    url = f"{SCHOLAR_BASE}/scholar?cluster={cluster_id}&hl=en"
    dbg.trace("scholar.fetch_bibtex", cluster=cluster_id)
    async with _ctx_or_new(headless=headless) as ctx:
        page = await ctx.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await _await_no_captcha(page)
            try:
                await page.wait_for_selector("div.gs_r", timeout=10_000)
            except Exception:
                dbg.trace("scholar.fetch_bibtex", "WARN: no gs_r row on cluster page")
            await asyncio.sleep(0.4)

            # The cite icon for each result has class gs_or_cit (anchor with onclick).
            cite_link = page.locator("a.gs_or_cit").first
            try:
                await cite_link.wait_for(timeout=10_000)
            except PWTimeout as e:
                raise RuntimeError(
                    f"Could not find Cite link for cluster {cluster_id}; Scholar layout may have changed."
                ) from e
            await cite_link.click()

            # Cite modal contains a BibTeX link.
            bibtex_link = page.locator("a:has-text('BibTeX')").first
            await bibtex_link.wait_for(timeout=10_000)

            href = await bibtex_link.get_attribute("href")
            if not href:
                raise RuntimeError("BibTeX link had no href.")
            if href.startswith("/"):
                href = SCHOLAR_BASE + href

            bib_page = await ctx.new_page()
            try:
                await bib_page.goto(href, wait_until="domcontentloaded")
                await _await_no_captcha(bib_page)
                text = await bib_page.inner_text("body")
                return text.strip()
            finally:
                await bib_page.close()
        finally:
            await page.close()


# Convenience sync wrappers ---------------------------------------------------


def search_sync(query: str, **kwargs) -> list[PaperHit]:
    return asyncio.run(search(query, **kwargs))


def fetch_versions_sync(versions_url: str, **kwargs) -> list[PaperHit]:
    return asyncio.run(fetch_versions(versions_url, **kwargs))


def fetch_bibtex_for_cluster_sync(cluster_id: str, **kwargs) -> str:
    return asyncio.run(fetch_bibtex_for_cluster(cluster_id, **kwargs))
