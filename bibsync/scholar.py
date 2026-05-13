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
            viewport={"width": 1280, "height": 900},
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )
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
        try:
            yield ctx
        finally:
            _SHARED_CTX.reset(token)


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

    Uses regex rather than a full HTML parser so we don't need BeautifulSoup as a dep,
    and the structure of Scholar result rows is stable enough for this.
    """
    hits: list[PaperHit] = []

    # Each result is a div.gs_r.gs_or.gs_scl with a data-cid (cluster id).
    for m in re.finditer(
        r'<div class="gs_r gs_or gs_scl"[^>]*?data-cid="([^"]+)"[^>]*>(.*?)</div>\s*</div>\s*</div>',
        html,
        flags=re.DOTALL,
    ):
        cluster_id = m.group(1)
        block = m.group(2)

        title_m = re.search(r'<h3 class="gs_rt"[^>]*>(.*?)</h3>', block, flags=re.DOTALL)
        if not title_m:
            continue
        title_html = title_m.group(1)
        title = re.sub(r"<[^>]+>", "", title_html)
        title = re.sub(r"\s+", " ", title).replace("[PDF]", "").replace("[HTML]", "").strip()
        title = title.lstrip("[").rstrip("]").strip()

        # gs_a: "Author1, Author2 - Venue, Year - publisher.com"
        authors_block_m = re.search(r'<div class="gs_a"[^>]*>(.*?)</div>', block, flags=re.DOTALL)
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
                # "Venue, Year" or just "Year"
                middle = parts[1]
                year_m = re.search(r"\b(19|20)\d{2}\b", middle)
                if year_m:
                    year = int(year_m.group(0))
                venue_part = re.sub(r",?\s*\b(19|20)\d{2}\b", "", middle).strip().strip(",")
                venue = venue_part or None

        cited_by = 0
        cited_m = re.search(r'Cited by (\d+)', block)
        if cited_m:
            cited_by = int(cited_m.group(1))

        versions_url: Optional[str] = None
        versions_m = re.search(r'href="(/scholar\?cluster=[^"]+)"[^>]*>\s*All\s*(\d+)\s*versions', block)
        if versions_m:
            versions_url = SCHOLAR_BASE + versions_m.group(1).replace("&amp;", "&")

        snippet_m = re.search(r'<div class="gs_rs"[^>]*>(.*?)</div>', block, flags=re.DOTALL)
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
        return "Scholar reports no matching articles."
    if any(m in low for m in ("captcha", "unusual traffic", "/sorry/", "recaptcha")):
        return (
            "Scholar is showing a CAPTCHA / bot-check page. Solve it in the browser "
            "window and re-run, or `bibsync config` then wipe the profile if it persists."
        )
    if "gs_r" not in low:
        return (
            "Scholar returned HTML but no expected result containers — your IP may be "
            "rate-limited or the page layout changed. Try again in a few minutes."
        )
    return f"Empty result for {query!r} (no obvious blocker detected)."


async def search(query: str, *, headless: bool = False, max_results: int = 10) -> list[PaperHit]:
    """Search Google Scholar for ``query`` and return parsed hits (top page only)."""
    url = f"{SCHOLAR_BASE}/scholar?q={urllib.parse.quote(query)}&hl=en"
    async with _ctx_or_new(headless=headless) as ctx:
        page = await ctx.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await _await_no_captcha(page)
            html = await page.content()
            hits = _parse_search_results(html)[:max_results]
            if not hits:
                # One-line diagnostic so the user knows what kind of empty this is.
                print(f"[bibsync] {_diagnose_empty_result(html, query)}")
            return hits
        finally:
            await page.close()


async def fetch_versions(versions_url: str, *, headless: bool = False) -> list[PaperHit]:
    """Fetch the 'All N versions' page for a paper cluster."""
    async with _ctx_or_new(headless=headless) as ctx:
        page = await ctx.new_page()
        try:
            await page.goto(versions_url, wait_until="domcontentloaded")
            await _await_no_captcha(page)
            html = await page.content()
            return _parse_search_results(html)
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
    async with _ctx_or_new(headless=headless) as ctx:
        page = await ctx.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await _await_no_captcha(page)

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
