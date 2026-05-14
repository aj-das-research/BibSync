"""BibSync command-line interface."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from . import (
    __version__,
    audit as audit_mod,
    bibtex,
    config as cfg,
    dbg,
    extract as extract_mod,
    fix as fix_mod,
    llm as llm_mod,
    picker,
    repair as repair_mod,
    scanner,
    scholar,
    suggest as suggest_mod,
    verify,
)
from .models import PaperHit

console = Console()


# Shared options ---------------------------------------------------------------

_bib_option = click.option(
    "--bib",
    "bib_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("references.bib"),
    show_default=True,
    help="Path to the .bib file to read/write.",
)
_headless_option = click.option(
    "--headless/--no-headless",
    default=False,
    show_default=True,
    help="Run the browser headlessly. Off by default because Google Scholar CAPTCHAs headless sessions.",
)
_use_openai_option = click.option(
    "--use-openai",
    is_flag=True,
    default=False,
    help="Use the OpenAI API to break ties when canonical-version selection is ambiguous. Requires OPENAI_API_KEY.",
)
_model_option = click.option(
    "--model",
    "--openai-model",
    "openai_model",
    default=None,
    help="LLM model override. If omitted, uses `llm_model` from config, or the provider default "
    "(gpt-4o-mini for OpenAI, openai/gpt-4o-mini for OpenRouter).",
)


@click.group()
@click.version_option(__version__, prog_name="bibsync")
@click.option(
    "--debug",
    is_flag=True,
    envvar="BIBSYNC_DEBUG",
    help="Emit per-step pipeline traces to stderr — Scholar searches, heuristic "
    "filter outcomes, LLM verdicts, candidate orderings. Useful when reporting bugs.",
)
def main(debug: bool) -> None:
    """BibSync — automate Google Scholar BibTeX and reconcile citations in LaTeX projects."""
    if debug:
        dbg.enable()
        console.print("[dim][debug] tracing enabled — events go to stderr[/dim]")


# add --------------------------------------------------------------------------


@main.command()
@click.argument("title", nargs=-1, required=True)
@_bib_option
@_headless_option
@_use_openai_option
@_model_option
@click.option(
    "--auto/--interactive",
    default=False,
    help="Auto-pick the top result without prompting (good for scripts).",
)
@click.option(
    "--all-versions/--no-all-versions",
    default=True,
    show_default=True,
    help="When the top result has multiple versions, fetch and consider all of them.",
)
def add(
    title: tuple[str, ...],
    bib_path: Path,
    headless: bool,
    use_openai: bool,
    openai_model: str,
    auto: bool,
    all_versions: bool,
) -> None:
    """Search Google Scholar for TITLE, pick the canonical version, append BibTeX to the .bib file."""
    query = " ".join(title).strip()
    if not query:
        console.print("[red]No title provided.[/red]")
        sys.exit(2)

    console.print(f"[cyan]Searching Google Scholar:[/cyan] {query!r}")
    hits = scholar.search_sync(query, headless=headless, max_results=10)
    if not hits:
        console.print("[red]No results found.[/red]")
        sys.exit(1)

    # Pick which hit to act on.
    selected = _select_hit(hits, auto=auto)
    if selected is None:
        console.print("[yellow]Aborted.[/yellow]")
        return

    # Expand to all versions if available.
    candidates = [selected]
    if all_versions and selected.versions_url:
        console.print("[cyan]Fetching all versions…[/cyan]")
        try:
            versions = scholar.fetch_versions_sync(selected.versions_url, headless=headless)
            if versions:
                candidates = versions
        except Exception as e:
            console.print(f"[yellow]Could not fetch versions ({e}); using top result.[/yellow]")

    if len(candidates) > 1:
        _show_candidates_table(candidates)

    canonical = (
        picker.pick_canonical_with_llm(candidates, model=openai_model)
        if use_openai and picker.is_ambiguous(candidates)
        else picker.pick_canonical(candidates)
    )
    console.print(f"[green]Canonical version:[/green] {canonical.short()}")

    # LLM verification — does this Scholar hit actually match the user's intended title?
    # Catches the "Attention Is All You Need" → "Is Attention All You Need?" by Mineault
    # case where Scholar returns only title-similar derivatives of the real paper.
    llm_cfg = cfg.resolve_llm_config()
    if llm_cfg is not None:
        verdict = llm_mod.verify_match(
            {"title": query, "author": "", "year": ""},
            canonical,
            model=openai_model,
            api_key=llm_cfg.api_key,
        )
        if not verdict.same_paper or verdict.confidence < 0.7:
            console.print(
                f"[yellow]⚠ LLM thinks this may not match your query[/yellow] "
                f"[dim](conf={verdict.confidence:.2f})[/dim]"
            )
            console.print(f"  [dim]Reasoning:[/dim] {verdict.reasoning}")
            if not auto:
                if not click.confirm("Add this entry anyway?", default=False):
                    console.print("[yellow]Aborted.[/yellow]")
                    return
            else:
                console.print("[red]Refusing to auto-add a low-confidence match. Re-run without --auto to override.[/red]")
                sys.exit(1)

    if not canonical.cluster_id:
        console.print("[red]Selected hit has no cluster id — cannot fetch BibTeX.[/red]")
        sys.exit(1)

    console.print("[cyan]Fetching BibTeX from Scholar…[/cyan]")
    try:
        bib_text = scholar.fetch_bibtex_for_cluster_sync(
            canonical.cluster_id, headless=headless
        )
    except Exception as e:
        console.print(f"[red]Failed to fetch BibTeX: {e}[/red]")
        sys.exit(1)

    new_db = bibtex.parse_string(bib_text)
    if not new_db.entries:
        console.print("[red]Scholar returned BibTeX but it failed to parse.[/red]")
        console.print(bib_text)
        sys.exit(1)
    new_entry = new_db.entries[0]

    existing_db = bibtex.load(bib_path)
    stored, was_added = bibtex.append_entry(existing_db, new_entry)
    if not was_added:
        console.print(
            f"[yellow]Already in {bib_path} as[/yellow] [bold]{stored['ID']}[/bold] — not duplicated."
        )
        return

    bibtex.dump(existing_db, bib_path)
    console.print(
        f"[green]Appended[/green] [bold]@{new_entry.get('ENTRYTYPE','misc')}{{{stored['ID']}}}[/bold] to {bib_path}"
    )


# suggest ----------------------------------------------------------------------


@main.command(name="suggest")
@click.argument("tex_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_bib_option
@_headless_option
@_model_option
@click.option(
    "--auto",
    is_flag=True,
    default=False,
    help="Accept every LLM suggestion without prompting (DANGEROUS — review your .tex after).",
)
@click.option(
    "--delay",
    type=float,
    default=1.5,
    show_default=True,
    help="Seconds to sleep between Scholar lookups.",
)
def suggest_cmd(
    tex_file: Path,
    bib_path: Path,
    headless: bool,
    openai_model: Optional[str],
    auto: bool,
    delay: float,
) -> None:
    """Read TEX_FILE, find paragraphs without citations, suggest + insert appropriate ones.

    For each paragraph that lacks any `\\cite{}` call, the LLM identifies what should be
    cited (named methods, foundational works, attributed claims). For each suggestion,
    BibSync searches Google Scholar, picks the canonical version, fetches BibTeX,
    appends to the .bib, and inserts a `\\cite{key}` into the .tex right after the
    relevant phrase.

    The .tex and .bib are both modified — review the diff before committing.
    """
    llm_cfg = cfg.resolve_llm_config()
    if not llm_cfg:
        console.print(
            "[red]No LLM API key found.[/red] Set one with: "
            "[bold]bibsync config set openrouter_key sk-or-...[/bold] or "
            "[bold]bibsync config set openai_key sk-...[/bold]"
        )
        sys.exit(2)
    console.print(f"[dim]Using {llm_cfg.provider} ({llm_cfg.model})[/dim]")

    # Closure state:
    #   suggestion_n           — running counter shown in the prompt header
    #   auto_accept_remaining  — once flipped True (via "a" choice), every subsequent
    #                            suggestion is auto-approved without prompting.
    suggestion_n = [0]
    auto_accept_remaining = [False]

    def approve(result, entry: dict) -> bool:
        suggestion_n[0] += 1
        scholar_hit = result.scholar_hit
        if auto_accept_remaining[0]:
            # Already in "yes to all remaining" mode — print a one-line confirmation
            # so the trace is still readable, then auto-accept.
            console.print(
                f"[dim]Suggestion #{suggestion_n[0]} · auto-accept · "
                f"{result.anchor!r} → \\cite{{{result.cite_key}}}[/dim]"
            )
            return True

        console.print()
        console.print(
            f"[bold]Suggestion #{suggestion_n[0]} · Paragraph {result.paragraph_index}[/bold]"
        )
        console.print(f"  [cyan]Anchor:[/cyan] {result.anchor!r}")
        ident = getattr(result, "identification", None)
        if ident and ident.expected_title:
            year_str = str(ident.expected_year) if ident.expected_year else "?"
            console.print(
                f"  [magenta]LLM identified:[/magenta] {ident.expected_title!r}"
                f" — {ident.expected_first_author or '?'} {year_str}"
                f" [dim]({ident.expected_venue or '?'}, conf={ident.confidence:.2f})[/dim]"
            )
        console.print(f"  [cyan]Query:[/cyan]  {result.query}")
        console.print(f"  [cyan]Reason:[/cyan] {result.reason}")
        if scholar_hit:
            console.print(f"  [green]Scholar match:[/green] {scholar_hit.short()}")
        console.print(f"  [yellow]Will insert:[/yellow] \\cite{{{result.cite_key}}}")
        choice = Prompt.ask(
            "  Accept? [y]es / [n]o / [a]ccept all remaining / [q]uit",
            choices=["y", "n", "a", "q"],
            default="y",
        )
        if choice == "q":
            raise click.Abort()
        if choice == "a":
            auto_accept_remaining[0] = True
            console.print(
                "[dim](auto-accepting all subsequent suggestions for this run)[/dim]"
            )
            return True
        return choice == "y"

    report = suggest_mod.suggest_for_file_sync(
        tex_file,
        bib_path,
        headless=headless,
        model=openai_model,
        api_key=llm_cfg.api_key,
        auto_approve=auto,
        approve_fn=None if auto else approve,
        delay_seconds=delay,
    )

    t = Table(title=f"Suggestion report — {tex_file}", show_lines=True)
    t.add_column("Para", justify="right")
    t.add_column("Status")
    t.add_column("Cite key", style="bold")
    t.add_column("Query")
    t.add_column("Note")
    status_styles = {
        "added": "[green]added[/green]",
        "skipped": "[yellow]skipped[/yellow]",
        "duplicate": "[dim]duplicate[/dim]",
        "no_scholar_hit": "[yellow]no hit[/yellow]",
        "anchor_not_found": "[yellow]anchor missing[/yellow]",
        "error": "[red]error[/red]",
    }
    for r in report.results:
        t.add_row(
            str(r.paragraph_index),
            status_styles.get(r.status, r.status),
            r.cite_key or "—",
            r.query,
            r.note,
        )
    console.print(t)
    console.print(
        f"[dim]{report.paragraphs_scanned} paragraphs scanned, "
        f"{report.paragraphs_with_existing_cites} already had citations.[/dim]"
    )
    console.print(
        "[bold]Summary:[/bold] " + ", ".join(f"{k}={v}" for k, v in report.summary().items())
    )


# fix --------------------------------------------------------------------------


@main.command(name="fix")
@_bib_option
@click.option(
    "--project",
    "project_root",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=None,
    help="Project root to scan for .tex files. If omitted, no .tex propagation is done.",
)
@_headless_option
@_model_option
@click.option(
    "--preserve-keys",
    is_flag=True,
    default=False,
    help="Keep the original cite keys instead of regenerating from corrected metadata. "
    "By default keys are rebuilt from author+year+title, and any \\cite{old} → \\cite{new} "
    "renames are propagated to every .tex file under --project.",
)
@click.option(
    "--delay",
    type=float,
    default=1.5,
    show_default=True,
    help="Seconds to sleep between Scholar lookups.",
)
def fix_cmd(
    bib_path: Path,
    project_root: Optional[Path],
    headless: bool,
    openai_model: Optional[str],
    preserve_keys: bool,
    delay: float,
) -> None:
    """Verify and rewrite a .bib by re-fetching every entry from Google Scholar.

    For each entry:
      1. Search Scholar by title.
      2. Filter candidates by author/year plausibility.
      3. Ask the LLM to verify each plausible candidate is the SAME paper.
      4. If verified, fetch the canonical BibTeX and merge Scholar's fields in.
      5. Rebuild the cite key from the corrected metadata (e.g. wang2019 → wang2021)
         and propagate \\cite{old} → \\cite{new} across every .tex under --project.

    Entries that fail any check are reported as 'unverified' and left untouched.
    """
    if not bib_path.exists():
        console.print(f"[red]{bib_path} not found.[/red]")
        sys.exit(1)

    llm_cfg = cfg.resolve_llm_config()
    if not llm_cfg:
        console.print(
            "[red]No LLM API key found.[/red] `fix` uses LLM verification to prevent "
            "wrong matches. Set a key with:\n"
            "  [bold]bibsync config set openrouter_key sk-or-...[/bold]\n"
            "  [bold]bibsync config set openai_key sk-...[/bold]"
        )
        sys.exit(2)
    console.print(
        f"[dim]Using {llm_cfg.provider} ({llm_cfg.model}) for match verification[/dim]"
    )

    regenerate_keys = not preserve_keys
    if regenerate_keys and not project_root:
        console.print(
            "[yellow]Cite keys may be regenerated from corrected metadata, but no --project "
            "was given. \\cite{} usages in .tex files won't be updated. "
            "Pass --project to keep them in sync, or --preserve-keys to keep keys as-is.[/yellow]"
        )

    report = fix_mod.fix_bib_sync(
        bib_path,
        project_root=project_root,
        headless=headless,
        regenerate_keys=regenerate_keys,
        model=openai_model,
        api_key=llm_cfg.api_key,
        delay_seconds=delay,
    )

    t = Table(title=f"Fix report — {bib_path}", show_lines=True)
    t.add_column("Original key", style="bold")
    t.add_column("New key")
    t.add_column("Status")
    t.add_column("Sim", justify="right")
    t.add_column("Scholar candidate", overflow="fold")
    t.add_column("LLM verdict")
    t.add_column("Field changes / note")
    status_styles = {
        "unchanged": "[dim]unchanged[/dim]",
        "rewritten": "[green]rewritten[/green]",
        "key_renamed": "[cyan]renamed[/cyan]",
        "unverified": "[yellow]unverified[/yellow]",
        "error": "[red]error[/red]",
    }
    for r in report.results:
        changes = "\n".join(r.field_changes) if r.field_changes else (r.note or "")
        new_key = r.new_key if r.new_key != r.original_key else "—"
        if r.llm_reasoning:
            verdict = f"[dim]conf={r.llm_confidence:.2f}[/dim]\n{r.llm_reasoning}"
        else:
            verdict = "—"
        if r.scholar_hit:
            authors = ", ".join(r.scholar_hit.authors[:2])
            if len(r.scholar_hit.authors) > 2:
                authors += " et al."
            candidate_cell = (
                f"{r.scholar_hit.title}\n"
                f"[dim]{authors} · {r.scholar_hit.year or '?'} · cited={r.scholar_hit.cited_by}[/dim]"
            )
        else:
            candidate_cell = "—"
        t.add_row(
            r.original_key,
            new_key,
            status_styles.get(r.status, r.status),
            f"{r.title_similarity:.0f}" if r.title_similarity else "—",
            candidate_cell,
            verdict,
            changes,
        )
    console.print(t)
    console.print(
        "[bold]Summary:[/bold] " + ", ".join(f"{k}={v}" for k, v in report.summary().items())
    )

    if report.tex_summary:
        s = report.tex_summary
        if s.files_changed:
            console.print(
                f"[green]Propagated key renames to {s.files_changed}/{s.files_scanned} .tex files.[/green]"
            )
            for f, n in s.edits_by_file.items():
                console.print(f"  • {f}: {n} edit(s)")
        else:
            console.print(f"[dim]No .tex files needed updating ({s.files_scanned} scanned).[/dim]")
    elif report.renames() and not project_root:
        console.print(
            "[yellow]Some keys were renamed in the .bib. Pass --project to propagate "
            "the renames to your .tex files.[/yellow]"
        )


# verify -----------------------------------------------------------------------


@main.command(name="verify")
@_bib_option
@_headless_option
@click.option(
    "--delay",
    type=float,
    default=2.0,
    show_default=True,
    help="Seconds to sleep between Scholar queries — be polite, avoid rate limits.",
)
def verify_cmd(bib_path: Path, headless: bool, delay: float) -> None:
    """Re-check each entry in the .bib against Google Scholar; flag discrepancies."""
    if not bib_path.exists():
        console.print(f"[red]{bib_path} not found.[/red]")
        sys.exit(1)

    db = bibtex.load(bib_path)
    if not db.entries:
        console.print(f"[yellow]{bib_path} has no entries.[/yellow]")
        return

    console.print(f"[cyan]Verifying {len(db.entries)} entries from {bib_path}…[/cyan]")
    results = verify.verify_entries_sync(
        db.entries, headless=headless, delay_seconds=delay
    )

    table = Table(title=f"Verification report — {bib_path}", show_lines=True)
    table.add_column("Entry", style="bold")
    table.add_column("Status")
    table.add_column("Similarity", justify="right")
    table.add_column("Issues")

    counts = {"verified": 0, "discrepancy": 0, "not_found": 0, "no_title": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        status_styles = {
            "verified": "[green]verified[/green]",
            "discrepancy": "[yellow]discrepancy[/yellow]",
            "not_found": "[red]not found[/red]",
            "no_title": "[red]no title[/red]",
        }
        issues = (
            "\n".join(
                f"{d.field}: bib={d.bib_value!r} ≠ scholar={d.scholar_value!r}"
                for d in r.discrepancies
            )
            or r.note
        )
        table.add_row(
            r.entry_id,
            status_styles.get(r.status, r.status),
            f"{r.title_similarity:.0f}" if r.title_similarity else "—",
            issues,
        )
    console.print(table)
    console.print(
        f"[bold]Summary:[/bold] "
        f"[green]{counts.get('verified', 0)} verified[/green], "
        f"[yellow]{counts.get('discrepancy', 0)} discrepancies[/yellow], "
        f"[red]{counts.get('not_found', 0)} not found[/red], "
        f"[red]{counts.get('no_title', 0)} missing title[/red]"
    )


# audit ------------------------------------------------------------------------


@main.command(name="audit")
@click.argument(
    "project_root",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=Path("."),
)
@_bib_option
@_model_option
@click.option(
    "--tier",
    type=click.IntRange(0, 2),
    default=1,
    show_default=True,
    help="Evidence depth for each citation: "
    "0 = metadata only (fastest, catches gross topic mismatches), "
    "1 = also fetch the paper's abstract from arXiv/Semantic Scholar/Crossref "
    "(catches misattributions where the title is on-topic but the actual "
    "contribution differs), "
    "2 = also download the open-access PDF, chunk + embed it, and retrieve the "
    "passages most relevant to each claim (catches specific numerical / "
    "factual mismatches; requires `pip install -e \".[audit-rag]\"` which "
    "includes pypdf + fastembed for fully-offline local embeddings).",
)
@click.option(
    "--rag-top-k",
    type=int,
    default=5,
    show_default=True,
    help="Tier 2 only: number of top-similarity chunks to retrieve per claim.",
)
@click.option(
    "--embedding-backend",
    type=click.Choice(["auto", "local", "api"]),
    default="auto",
    show_default=True,
    help="Tier 2 only: where to compute embeddings. "
    "'local' uses fastembed (BAAI/bge-small-en-v1.5 by default, fully offline, no API key); "
    "'api' uses the OpenAI-compatible embeddings endpoint of your configured LLM provider; "
    "'auto' tries local first, then API.",
)
@click.option(
    "--embedding-model",
    default="auto",
    show_default=True,
    help="Tier 2 only: model name for the embeddings call. 'auto' resolves "
    "per-backend: BAAI/bge-small-en-v1.5 for local; for the API backend it "
    "picks baai/bge-m3 on OpenRouter (sk-or-...) and text-embedding-3-small "
    "on OpenAI (sk-...). Override only if you need a specific model.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for paper-content / PDF / embedding caches. Defaults to the "
    "platform user-cache dir (e.g. ~/Library/Caches/bibsync on macOS).",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Bypass on-disk caches; re-fetch abstracts and re-embed chunks.",
)
@click.option(
    "--fix",
    "do_fix",
    is_flag=True,
    default=False,
    help="Replace hallucinated \\cite{key} occurrences with a marker comment "
    "(keeps the rest of the \\cite{...} list intact when only some keys are bad).",
)
@click.option(
    "--per-dir-bib",
    is_flag=True,
    default=False,
    help="For multi-subproject trees: for each .tex file, audit against the NEAREST "
    ".bib in its directory chain rather than the single --bib file. Use when the "
    "project has multiple subdirectories each with their own bibliography.",
)
@click.option(
    "--confidence-floor",
    type=float,
    default=0.7,
    show_default=True,
    help="Only mark a citation as hallucinated when the LLM's supports=false "
    "verdict is at least this confident. Weaker no-support verdicts fall into "
    "'unverifiable' instead.",
)
@click.option(
    "--delay",
    type=float,
    default=0.5,
    show_default=True,
    help="Seconds to sleep between LLM calls (be polite to your API quota).",
)
def audit_cmd(
    project_root: Path,
    bib_path: Path,
    openai_model: Optional[str],
    tier: int,
    rag_top_k: int,
    embedding_backend: str,
    embedding_model: str,
    cache_dir: Optional[Path],
    no_cache: bool,
    do_fix: bool,
    per_dir_bib: bool,
    confidence_floor: float,
    delay: float,
) -> None:
    """Audit every \\cite{} in PROJECT_ROOT — verify each cited paper actually
    supports the claim it's attached to.

    For each \\cite{key}: looks up the .bib entry, then asks the LLM whether the
    paper's topic/contribution supports the surrounding prose claim. Reports:
      * verified       — the cited paper supports the claim
      * hallucinated   — clear topic mismatch (likely LLM-fabricated citation)
      * unverifiable   — LLM couldn't decide confidently; keep as-is
      * missing_in_bib — cited key has no entry in the .bib (broken reference)

    Pass --fix to replace every confidently-hallucinated \\cite{} with a marker
    comment that records the original key and the LLM's reasoning, so you can
    see (and manually restore if needed) what was removed.
    """
    if not bib_path.exists():
        console.print(f"[red]{bib_path} not found.[/red]")
        sys.exit(1)

    llm_cfg = cfg.resolve_llm_config()
    if not llm_cfg:
        console.print(
            "[red]No LLM API key found.[/red] `audit` needs an LLM for claim verification. "
            "Set one with:\n  [bold]bibsync config set openrouter_key sk-or-...[/bold]"
        )
        sys.exit(2)
    tier_blurb = {
        0: "metadata only",
        1: "+ abstract from arXiv / Semantic Scholar / Crossref",
        2: "+ PDF chunks via RAG retrieval",
    }[tier]
    console.print(
        f"[dim]Using {llm_cfg.provider} ({llm_cfg.model}) "
        f"for citation audit — tier {tier} ({tier_blurb})[/dim]"
    )

    report = audit_mod.audit_project_sync(
        project_root,
        bib_path,
        tier=tier,
        model=openai_model,
        api_key=llm_cfg.api_key,
        delay_seconds=delay,
        fix=do_fix,
        confidence_floor=confidence_floor,
        cache_dir=cache_dir,
        no_cache=no_cache,
        rag_top_k=rag_top_k,
        embedding_model=embedding_model,
        embedding_backend=embedding_backend,
        per_dir_bib=per_dir_bib,
    )

    if not report.checks:
        console.print(f"[dim]No \\cite{{}} calls found in {project_root}.[/dim]")
        return

    # Pre-emptive UX guard: if a large fraction of cites come back missing_in_bib,
    # this is almost always a setup error (wrong --bib, or auditing a multi-subproject
    # tree without --per-dir-bib). Surface it loudly BEFORE the giant table.
    summary_check = report.summary()
    n_total = len(report.checks)
    n_missing = summary_check.get("missing_in_bib", 0)
    if n_total >= 3 and n_missing / n_total >= 0.5:
        bib_files_in_tree = []
        try:
            bib_files_in_tree = sorted(
                p
                for p in project_root.rglob("*.bib")
                if not any(part in {".git", ".venv", "node_modules"} for part in p.parts)
            )
        except OSError:
            pass
        console.print()
        console.print(
            "[bold red]⚠  Most citations came back as `missing_in_bib`.[/bold red] "
            f"({n_missing}/{n_total})"
        )
        console.print(
            "    This usually means the [bold]--bib[/bold] you passed doesn't contain the keys "
            "the .tex files reference."
        )
        if len(bib_files_in_tree) > 1 and not per_dir_bib:
            console.print(
                f"    Your project tree has [bold]{len(bib_files_in_tree)} .bib files[/bold] — "
                f"audit only used [bold]{bib_path}[/bold]."
            )
            console.print("    [green]Fix:[/green] re-run with [bold]--per-dir-bib[/bold] "
                          "to audit each .tex against its nearest .bib:")
            console.print(
                f"      [dim]bibsync audit {project_root} --per-dir-bib --tier {tier}"
                + (" --fix" if do_fix else "") + "[/dim]"
            )
            console.print("    …or scope the audit to a single subproject:")
            sample = bib_files_in_tree[0]
            console.print(
                f"      [dim]bibsync audit {sample.parent} --bib {sample} --tier {tier}[/dim]"
            )
        else:
            console.print("    [green]Check:[/green] does this --bib actually correspond to the "
                          "scanned .tex files?")
        console.print()

    t = Table(
        title=f"Citation audit — {project_root}  (bib: {bib_path.name})",
        show_lines=True,
    )
    t.add_column("Cite key", style="bold")
    t.add_column("Status")
    t.add_column("Evidence", justify="center")
    t.add_column("Location")
    t.add_column("Conf.", justify="right")
    t.add_column("Claim → paper", overflow="fold")
    t.add_column("LLM reasoning", overflow="fold")
    status_styles = {
        "verified": "[green]verified[/green]",
        "hallucinated": "[red]hallucinated[/red]",
        "unverifiable": "[yellow]unverifiable[/yellow]",
        "missing_in_bib": "[red]missing_in_bib[/red]",
    }
    tier_labels = {0: "meta", 1: "abs", 2: "RAG"}
    for c in report.checks:
        loc = f"{c.file.name}:{c.line}"
        if c.bib_entry:
            paper_summary = (c.bib_entry.get("title") or "").strip()
            paper_authors = (c.bib_entry.get("author") or "").split(" and ")[0].strip()
            if paper_authors:
                paper_summary += f" — {paper_authors}"
        else:
            paper_summary = "(not in .bib)"
        claim_preview = c.claim_text[:120] + ("…" if len(c.claim_text) > 120 else "")
        claim_vs_paper = f"[dim]claim:[/dim] {claim_preview}\n[dim]paper:[/dim] {paper_summary}"
        reasoning_cell = c.reasoning + ("\n[green](fixed)[/green]" if c.fixed else "")
        ev_label = tier_labels.get(c.evidence_tier, str(c.evidence_tier))
        if c.evidence_tier == 2 and c.n_chunks:
            ev_label = f"RAG×{c.n_chunks}"
        t.add_row(
            c.cite_key,
            status_styles.get(c.status, c.status),
            ev_label,
            loc,
            f"{c.confidence:.2f}" if c.confidence else "—",
            claim_vs_paper,
            reasoning_cell,
        )
    console.print(t)

    summary = report.summary()
    parts = []
    for status in ("verified", "hallucinated", "unverifiable", "missing_in_bib"):
        if status in summary:
            color = "green" if status == "verified" else (
                "red" if status in ("hallucinated", "missing_in_bib") else "yellow"
            )
            parts.append(f"[{color}]{summary[status]} {status}[/{color}]")
    console.print(f"[bold]Summary:[/bold] " + ", ".join(parts))

    # ── Tier-2 degradation summary ─────────────────────────────────────────
    # If the user asked for tier N but most cites only achieved a lower tier,
    # surface that loudly — otherwise the user sees "verified" verdicts that
    # were actually made on weaker evidence than they requested.
    if tier >= 1 and report.checks:
        auditable = [c for c in report.checks if c.status != "missing_in_bib"]
        if auditable:
            degraded = [c for c in auditable if c.evidence_tier < tier]
            if degraded and len(degraded) / len(auditable) >= 0.25:
                reason_counts: dict[str, int] = {}
                for c in degraded:
                    reason_counts[c.degraded_reason or "unknown"] = (
                        reason_counts.get(c.degraded_reason or "unknown", 0) + 1
                    )
                _REASON_HUMAN = {
                    "source_not_found": "no source (arXiv/SS/Crossref all missed or rate-limited)",
                    "no_open_access_pdf": "paper found but no open-access PDF",
                    "pdf_download_or_extract_failed": "PDF download/extract failed",
                    "embedding_failed": "embeddings unavailable (install fastembed or use --embedding-backend api)",
                    "unknown": "unknown reason — re-run with --debug to see traces",
                }
                lines = [
                    f"[bold yellow]⚠  Tier-{tier} requested, but {len(degraded)}/{len(auditable)} citation(s) "
                    f"verified on weaker evidence.[/bold yellow]"
                ]
                for reason, n in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
                    lines.append(f"   • [yellow]{n}[/yellow] × {_REASON_HUMAN.get(reason, reason)}")
                lines.append(
                    "[dim]Verdicts marked `verified` on degraded evidence may be "
                    "false-positives. Re-run with --debug to see why.[/dim]"
                )
                console.print()
                for ln in lines:
                    console.print(ln)

    fixed_count = sum(1 for c in report.checks if c.fixed)
    if fixed_count:
        console.print(
            f"[green]Applied {fixed_count} fix(es)[/green] — review the .tex diff "
            f"before committing."
        )
    elif do_fix and summary.get("hallucinated", 0) == 0:
        console.print("[dim]--fix was set but no hallucinated citations were found.[/dim]")


# scan -------------------------------------------------------------------------


@main.command(name="scan")
@click.argument(
    "project_root",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=Path("."),
)
def scan_cmd(project_root: Path) -> None:
    """Scan a LaTeX PROJECT_ROOT for \\cite{} usages and reconcile with .bib files."""
    report = scanner.scan(project_root)
    console.print(
        f"[cyan]Scanned[/cyan] {len(report.tex_files)} .tex file(s), "
        f"{len(report.bib_files)} .bib file(s) under {report.project_root}"
    )
    console.print(
        f"  cited keys: {len(report.cited_keys)}    defined keys: {len(report.defined_keys)}"
    )

    if report.missing_keys:
        t = Table(title="Missing from .bib (cited but not defined — possible hallucinations)")
        t.add_column("Key", style="red bold")
        t.add_column("Used in")
        for key in sorted(report.missing_keys):
            uses = report.uses_of(key)
            t.add_row(
                key,
                "\n".join(f"{u.file.relative_to(report.project_root)}:{u.line}" for u in uses[:5]),
            )
        console.print(t)
    else:
        console.print("[green]No missing citation keys.[/green]")

    if report.orphan_keys:
        t = Table(title="Orphan entries (defined in .bib but never cited)")
        t.add_column("Key", style="yellow")
        for key in sorted(report.orphan_keys):
            t.add_row(key)
        console.print(t)
    else:
        console.print("[green]No orphan .bib entries.[/green]")


# search (preview-only convenience) -------------------------------------------


@main.command()
@click.argument("title", nargs=-1, required=True)
@_headless_option
def search(title: tuple[str, ...], headless: bool) -> None:
    """Search Google Scholar and print results without writing anything."""
    query = " ".join(title).strip()
    hits = scholar.search_sync(query, headless=headless, max_results=10)
    if not hits:
        console.print("[red]No results.[/red]")
        return
    _show_candidates_table(hits)


# config -----------------------------------------------------------------------


@main.group(name="config")
def config_group() -> None:
    """Manage BibSync configuration (e.g., OpenAI API key)."""


@config_group.command(name="path")
def config_path_cmd() -> None:
    """Show where the config file lives."""
    console.print(str(cfg.config_path()))


@config_group.command(name="reset-profile")
def config_reset_profile() -> None:
    """Wipe the persistent Chrome profile used by the Scholar scraper.

    Run this when Scholar has flagged your session and even solving the CAPTCHA
    doesn't restore results. After resetting, the next run starts fresh — you'll
    likely see a CAPTCHA on first search, solve it once, and proceed.
    """
    path = scholar.reset_profile()
    console.print(f"[green]Wiped Chrome profile at[/green] {path}")


@config_group.command(name="show")
def config_show() -> None:
    """Print the resolved config (with secrets redacted)."""
    data = cfg.load_config()
    if data:
        console.print("[bold]Stored config:[/bold]")
        for k, v in data.items():
            if "key" in k.lower() and isinstance(v, str) and len(v) > 8:
                v = v[:6] + "…" + v[-2:]
            console.print(f"  {k} = {v}")
    else:
        console.print("[dim]No stored config.[/dim]")

    resolved = cfg.resolve_llm_config()
    if resolved is None:
        console.print(
            "\n[bold]LLM:[/bold] [red]no API key found[/red] — set one with "
            "[bold]bibsync config set openrouter_key sk-or-...[/bold] or "
            "[bold]bibsync config set openai_key sk-...[/bold]"
        )
        return
    redacted = resolved.api_key[:6] + "…" + resolved.api_key[-2:]
    color = "magenta" if resolved.provider == "openrouter" else "cyan"
    console.print(f"\n[bold]Resolved LLM config:[/bold]")
    console.print(f"  provider : [{color}]{resolved.provider}[/{color}]")
    console.print(f"  api_key  : {redacted}   [dim](source: {resolved.source})[/dim]")
    console.print(f"  base_url : {resolved.base_url or '(openai default)'}")
    console.print(f"  model    : {resolved.model}")


@config_group.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a config key (e.g. `bibsync config set openai_key sk-...`)."""
    data = cfg.load_config()
    data[key] = value
    p = cfg.save_config(data)
    console.print(f"[green]Saved[/green] {key} → {p}")


@config_group.command(name="unset")
@click.argument("key")
def config_unset(key: str) -> None:
    """Remove a config key."""
    data = cfg.load_config()
    if key in data:
        del data[key]
        cfg.save_config(data)
        console.print(f"[green]Removed[/green] {key}")
    else:
        console.print(f"[yellow]{key} was not set.[/yellow]")


# extract ----------------------------------------------------------------------


@main.command(name="extract")
@click.argument("tex_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_bib_option
@_headless_option
@_model_option
@click.option(
    "--confidence-floor",
    type=float,
    default=extract_mod.DEFAULT_CONFIDENCE_FLOOR,
    show_default=True,
    help="Skip cite keys whose LLM inference confidence is below this threshold.",
)
@click.option(
    "--all/--only-missing",
    "include_all",
    default=False,
    show_default=True,
    help="By default, only resolve keys not already in the .bib. --all re-resolves everything.",
)
@click.option(
    "--delay",
    type=float,
    default=1.5,
    show_default=True,
    help="Seconds to sleep between Scholar lookups.",
)
def extract_cmd(
    tex_file: Path,
    bib_path: Path,
    headless: bool,
    openai_model: str,
    confidence_floor: float,
    include_all: bool,
    delay: float,
) -> None:
    """Resolve every \\cite{} in TEX_FILE: LLM-infers each key, fetches BibTeX, appends to .bib.

    Example: BibSync sees `\\cite{moor2023gmai}` in a paragraph about "Generalist medical AI",
    asks the LLM "what paper is `moor2023gmai`?", gets back "Moor et al. 2023 — Generalist
    Medical AI", searches Scholar, picks the canonical version, fetches BibTeX, appends.
    """
    llm_cfg = cfg.resolve_llm_config()
    if not llm_cfg:
        console.print(
            "[red]No LLM API key found.[/red] Set one with: "
            "[bold]bibsync config set openrouter_key sk-or-...[/bold] or "
            "[bold]bibsync config set openai_key sk-...[/bold]"
        )
        sys.exit(2)
    api_key = llm_cfg.api_key
    console.print(
        f"[dim]Using {llm_cfg.provider} ({llm_cfg.model})[/dim]"
    )

    console.print(f"[cyan]Extracting citations from[/cyan] {tex_file} → {bib_path}")
    report = extract_mod.extract_from_file_sync(
        tex_file,
        bib_path,
        only_missing=not include_all,
        headless=headless,
        confidence_floor=confidence_floor,
        openai_model=openai_model,
        api_key=api_key,
        delay_seconds=delay,
    )

    t = Table(title=f"Extraction report — {tex_file}", show_lines=True)
    t.add_column("Cite key", style="bold")
    t.add_column("Status")
    t.add_column("Inferred title")
    t.add_column("Conf.", justify="right")
    t.add_column("Note")
    status_styles = {
        "added": "[green]added[/green]",
        "duplicate": "[dim]duplicate[/dim]",
        "low_confidence": "[yellow]low confidence[/yellow]",
        "no_scholar_hit": "[yellow]no hit[/yellow]",
        "error": "[red]error[/red]",
    }
    for r in report.results:
        inferred_title = (r.inferred.title if r.inferred else "") or "—"
        conf = f"{r.inferred.confidence:.2f}" if r.inferred else "—"
        t.add_row(
            r.cite_key,
            status_styles.get(r.status, r.status),
            inferred_title,
            conf,
            r.note,
        )
    console.print(t)
    summary = report.summary()
    console.print(
        "[bold]Summary:[/bold] "
        + ", ".join(f"{k}={v}" for k, v in summary.items())
    )


# repair -----------------------------------------------------------------------


@main.command(name="repair")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--bib",
    "bib_output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional .bib file to append repaired entries to. If omitted, prints to stdout.",
)
@_headless_option
@_model_option
@click.option(
    "--delay",
    type=float,
    default=1.5,
    show_default=True,
    help="Seconds to sleep between Scholar lookups.",
)
def repair_cmd(
    source: Path,
    bib_output: Optional[Path],
    headless: bool,
    openai_model: str,
    delay: float,
) -> None:
    """Repair legacy \\bibitem{} entries in SOURCE into verified BibTeX.

    SOURCE is a .tex or .bbl file containing \\bibitem{key} blocks. BibSync LLM-parses
    each block, searches Scholar to verify, and emits corrected BibTeX.
    """
    llm_cfg = cfg.resolve_llm_config()
    if not llm_cfg:
        console.print(
            "[red]No LLM API key found.[/red] Set one with: "
            "[bold]bibsync config set openrouter_key sk-or-...[/bold] or "
            "[bold]bibsync config set openai_key sk-...[/bold]"
        )
        sys.exit(2)
    api_key = llm_cfg.api_key
    console.print(
        f"[dim]Using {llm_cfg.provider} ({llm_cfg.model})[/dim]"
    )

    console.print(f"[cyan]Repairing \\bibitem entries in[/cyan] {source}")
    report = repair_mod.repair_file_sync(
        source,
        bib_output=bib_output,
        headless=headless,
        openai_model=openai_model,
        api_key=api_key,
        delay_seconds=delay,
    )

    t = Table(title=f"Repair report — {source}", show_lines=True)
    t.add_column("Cite key", style="bold")
    t.add_column("Status")
    t.add_column("Scholar title")
    t.add_column("Discrepancies")
    status_styles = {
        "repaired": "[green]repaired[/green]",
        "discrepancy": "[yellow]repaired w/ discrepancy[/yellow]",
        "no_scholar_hit": "[yellow]no hit[/yellow]",
        "error": "[red]error[/red]",
    }
    for r in report.results:
        scholar_title = r.scholar_hit.title if r.scholar_hit else "—"
        disc = "\n".join(r.discrepancies) if r.discrepancies else (r.note or "")
        t.add_row(r.cite_key, status_styles.get(r.status, r.status), scholar_title, disc)
    console.print(t)
    summary = report.summary()
    console.print(
        "[bold]Summary:[/bold] " + ", ".join(f"{k}={v}" for k, v in summary.items())
    )

    if bib_output is None:
        # Print BibTeX to stdout for manual review.
        console.print("\n[bold]Repaired BibTeX (review before adding to your .bib):[/bold]")
        from bibtexparser.bibdatabase import BibDatabase
        from bibtexparser.bwriter import BibTexWriter
        import bibtexparser as bp

        db = BibDatabase()
        db.entries = [r.new_bibtex_entry for r in report.results if r.new_bibtex_entry]
        if db.entries:
            w = BibTexWriter()
            w.indent = "  "
            console.print(bp.dumps(db, writer=w))
    else:
        console.print(f"[green]Appended[/green] repaired entries to {bib_output}")


# Helpers ----------------------------------------------------------------------


def _show_candidates_table(hits: list[PaperHit]) -> None:
    t = Table(show_lines=False)
    t.add_column("#", style="dim", justify="right")
    t.add_column("Title")
    t.add_column("Year", justify="right")
    t.add_column("Venue")
    t.add_column("Cited", justify="right")
    t.add_column("Versions", justify="right")
    for i, h in enumerate(hits):
        t.add_row(
            str(i),
            h.title,
            str(h.year or "—"),
            h.venue or "—",
            str(h.cited_by),
            "yes" if h.versions_url else "—",
        )
    console.print(t)


def _select_hit(hits: list[PaperHit], *, auto: bool) -> Optional[PaperHit]:
    if auto or len(hits) == 1:
        return hits[0]
    _show_candidates_table(hits)
    choice = Prompt.ask(
        "Pick a result by index (or 'q' to abort)", default="0"
    )
    if choice.strip().lower() in {"q", "quit", "abort"}:
        return None
    try:
        idx = int(choice)
        if 0 <= idx < len(hits):
            return hits[idx]
    except ValueError:
        pass
    console.print("[red]Invalid choice.[/red]")
    return None


if __name__ == "__main__":
    main()
