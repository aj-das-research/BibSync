"""``bibsync serve`` — local HTTP+JSON server that drives the AI from
non-CLI clients (Chrome extension, VS Code plugin, CI, scripts).

Design contracts:

  • **Local-only.** Binds to ``127.0.0.1`` by default; never accept an
    external bind without an explicit ``--host 0.0.0.0`` (which the CLI
    refuses unless ``BIBSYNC_ALLOW_EXTERNAL=1`` env var is set).

  • **Token auth.** Every request must include
    ``Authorization: Bearer <token>``. The token is regenerated on
    each ``bibsync serve`` launch and written to
    ``~/.config/bibsync/server.token``. Clients read the token from
    that file at startup.

  • **In-memory contract.** Endpoints take tex/bib content INLINE in
    JSON request bodies — they don't read from the filesystem. The
    caller (extension) is responsible for shipping the manuscript
    content and writing the response back to its host (Overleaf, an
    editor, …). The server never touches the user's filesystem
    EXCEPT to read/write its own caches at ``~/Library/Caches/bibsync/``.

  • **Atomic patches.** Edits flow through ``patches.apply_patches``;
    all-or-nothing semantics, conflict detection on stale ``old_text``.

  • **Thin wrappers.** Each endpoint calls one ``*_sync`` entry point
    in the audit / suggest / evidence / source_rank module and
    returns its ``to_dict()``. No new core logic lives in the
    server — it's purely a network surface.

Endpoints (Sprint-D scope):

  GET  /health                — connectivity + version + cache stats
  POST /audit                 — verify every \\cite{} in supplied tex/bib
  POST /suggest               — find + propose missing citations
  POST /evidence              — evidence retrieval for a free-form claim
  POST /source-rank           — rank canonical candidates for a claim/title
  POST /patch/preview         — render diff without applying
  POST /patch/apply           — atomic application with conflict detection
  GET  /memory                — list memory records (filterable)
  POST /memory/forget         — write tombstone
  DELETE /memory/project      — purge_project equivalent
  GET  /cache/status          — sizes of paper_content / pdfs / embeddings
  POST /cache/clear           — clear optional caches (memory preserved)
  GET  /privacy               — what data is held for the requesting project
  GET  /openapi.json          — FastAPI-auto OpenAPI spec
"""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from . import __version__, config as cfg, dbg

# Module-level Pydantic schemas — FastAPI's body-vs-query auto-detection only
# fires reliably for BaseModel subclasses at module scope. Defining them
# inside ``create_app`` caused FastAPI to treat them as query parameters.
try:
    from pydantic import BaseModel, Field

    class AuditRequest(BaseModel):
        project_id: str = ""
        tex_files: dict = Field(default_factory=dict)
        bib_files: dict = Field(default_factory=dict)
        tier: int = 2
        rag_top_k: int = 5
        embedding_backend: str = "auto"
        use_memory: bool = True

    class EvidenceRequest(BaseModel):
        claim: str
        top_papers: int = 5
        tier: int = 2
        rag_top_k: int = 5
        embedding_backend: str = "auto"

    class SourceRankRequest(BaseModel):
        query: str
        top_papers: int = 5

    class PatchPreviewRequest(BaseModel):
        patches: list = Field(default_factory=list)
        files: dict = Field(default_factory=dict)

    class PatchApplyRequest(BaseModel):
        patches: list = Field(default_factory=list)
        files: dict = Field(default_factory=dict)
        require_approval: bool = True

    class MemoryForgetRequest(BaseModel):
        record_id: str
        scope: str = "project"
        project_root: Optional[str] = None
        project_id: Optional[str] = None

    class MemoryRememberRequest(BaseModel):
        project_id: Optional[str] = None
        project_root: Optional[str] = None
        type: str                       # accept | reject | verdict | preference | override
        claim_text: str = ""
        paper_key: str = ""
        cite_key: str = ""
        decision: str = ""
        tier: int = 0
        confidence: float = 0.0
        source: str = "extension"
        rationale: str = ""
        scope: str = "project"

    class MemoryPurgeRequest(BaseModel):
        project_root: str

    _PYDANTIC_OK = True
except ImportError:
    _PYDANTIC_OK = False

# ── token management ────────────────────────────────────────────────────────

_TOKEN_FILENAME = "server.token"


def _token_path() -> Path:
    """Per-user token file. Same dir as the rest of bibsync's config."""
    from platformdirs import user_config_dir
    return Path(user_config_dir("bibsync", "bibsync")) / _TOKEN_FILENAME


def generate_token() -> str:
    """Generate a fresh per-process token + write it to ``_token_path``.

    The file is mode 0600 so other users on a multi-user system can't
    read it. The token is opaque hex of ~32 bytes — overwhelmingly
    longer than necessary for the local-only auth surface but cheap.
    """
    token = secrets.token_urlsafe(32)
    p = _token_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token, encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    dbg.trace("server.token", "generated", path=str(p))
    return token


# ── helper: write request files to a temp dir + run audit ───────────────────


async def _audit_inline(
    *,
    tex_files: dict,
    bib_files: dict,
    tier: int = 2,
    rag_top_k: int = 5,
    embedding_backend: str = "auto",
    use_memory: bool = True,
    project_id: str = "",
    api_key: str,
) -> dict:
    """Write the inline tex/bib content to a temp directory and run
    ``audit_project`` against it. Returns the report's ``to_dict()``.

    The temp dir is short-lived; caches are SHARED with the user's
    persistent ``~/Library/Caches/bibsync/`` so memory recall works
    across server invocations.
    """
    from . import audit as audit_mod

    with TemporaryDirectory(prefix="bibsync-serve-") as td:
        root = Path(td)
        for fname, content in (tex_files or {}).items():
            p = root / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        bib_path: Optional[Path] = None
        for fname, content in (bib_files or {}).items():
            p = root / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            if bib_path is None:  # use the first bib as the primary
                bib_path = p
        # If no bib supplied, create an empty one so audit_project doesn't fail
        # on its first .bib lookup.
        if bib_path is None:
            bib_path = root / "references.bib"
            bib_path.write_text("", encoding="utf-8")

        report = await audit_mod.audit_project(
            root, bib_path,
            tier=tier, rag_top_k=rag_top_k,
            embedding_backend=embedding_backend,
            api_key=api_key, use_memory=use_memory,
            # Pass the client's stable project_id so memory persists
            # across calls — the temp `root` would otherwise give every
            # request a fresh, useless namespace.
            memory_project_id=project_id or None,
        )
    return report.to_dict()


# ── FastAPI app construction ────────────────────────────────────────────────


def create_app(*, token: Optional[str] = None):
    """Build the FastAPI app. ``token`` is the bearer token clients must
    present. When None, falls back to reading ``server.token`` (used by
    tests that bypass the CLI entry point)."""
    if not _PYDANTIC_OK:
        raise RuntimeError(
            "pydantic not installed. Install with: pip install -e \".[server]\""
        )
    try:
        from fastapi import FastAPI, Header, HTTPException, status
    except ImportError:
        raise RuntimeError(
            "FastAPI not installed. Install with: pip install -e \".[server]\""
        )

    # ── auth dependency ────────────────────────────────────────────────────
    # ``token`` is authoritative: None means NO auth (the --no-token default;
    # the server is 127.0.0.1-bound). We deliberately do NOT fall back to a
    # token file on disk — a stale file from a past --token run would
    # otherwise silently re-enable auth and reject the extension.
    expected_token = token

    def require_auth(authorization: Optional[str] = Header(None)) -> None:
        if not expected_token:
            return  # no token configured — open on localhost
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or malformed Authorization header",
            )
        provided = authorization[len("Bearer "):].strip()
        # Constant-time compare to avoid timing-attack signal.
        if not secrets.compare_digest(provided, expected_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="invalid bearer token",
            )

    # ── app + endpoints ────────────────────────────────────────────────────
    app = FastAPI(
        title="BibSync",
        version=__version__,
        description=(
            "Local-first citation AI server. The Chrome extension (an "
            "extension-origin page) fetches these endpoints directly. "
            "Auth is optional — when a token is configured, send "
            "Authorization: Bearer <token>."
        ),
    )

    # ── CORS + Private Network Access ──────────────────────────────────────
    # The Chrome extension's side panel is a chrome-extension:// page; it
    # fetches this server directly. Extension pages with host_permissions
    # bypass CORS, but a permissive policy here is cheap insurance and also
    # lets a plain browser tab / curl hit the server. Origins are wide open
    # because the server is 127.0.0.1-bound — the network layer is the real
    # boundary, not CORS.
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _allow_private_network(request, call_next):
        """Chrome's Private Network Access sends a preflight with
        ``Access-Control-Request-Private-Network: true`` when a page
        reaches a local-network address. Answer it so the extension's
        localhost fetches aren't blocked on newer Chrome."""
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    from fastapi import Depends as _Depends

    @app.get("/health", dependencies=[_Depends(require_auth)])
    def get_health() -> dict:
        """Connectivity + version probe. Cheap; safe to poll every 30s."""
        return {
            "ok": True,
            "version": __version__,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    @app.post("/audit", dependencies=[_Depends(require_auth)])
    async def post_audit(req: AuditRequest) -> dict:
        """Run the full audit pipeline on inline tex/bib content."""
        llm_cfg = cfg.resolve_llm_config()
        if not llm_cfg:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="No LLM API key configured")
        return await _audit_inline(
            tex_files=req.tex_files, bib_files=req.bib_files,
            tier=req.tier, rag_top_k=req.rag_top_k,
            embedding_backend=req.embedding_backend,
            use_memory=req.use_memory, project_id=req.project_id,
            api_key=llm_cfg.api_key,
        )

    @app.post("/evidence", dependencies=[_Depends(require_auth)])
    async def post_evidence(req: EvidenceRequest) -> dict:
        """Find evidence for a free-form claim. Reuses bibsync evidence."""
        from . import evidence_cmd as ev_mod
        llm_cfg = cfg.resolve_llm_config()
        if not llm_cfg:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="No LLM API key configured")
        report = await ev_mod.find_evidence_for_claim(
            req.claim,
            top_papers=req.top_papers, tier=req.tier,
            rag_top_k=req.rag_top_k,
            embedding_backend=req.embedding_backend,
            api_key=llm_cfg.api_key,
        )
        return report.to_dict()

    @app.post("/source-rank", dependencies=[_Depends(require_auth)])
    async def post_source_rank(req: SourceRankRequest) -> dict:
        """Rank canonical candidate papers for a claim/title."""
        from . import source_rank as sr_mod
        llm_cfg = cfg.resolve_llm_config()
        if not llm_cfg:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="No LLM API key configured")
        report = await sr_mod.rank_sources(
            req.query, top_papers=req.top_papers, api_key=llm_cfg.api_key,
        )
        return report.to_dict()

    @app.post("/patch/preview", dependencies=[_Depends(require_auth)])
    def post_patch_preview(req: PatchPreviewRequest) -> dict:
        """Render diff for the given patches without applying. Always safe."""
        from .patches import preview_patches
        return preview_patches(req.patches, req.files)

    @app.post("/patch/apply", dependencies=[_Depends(require_auth)])
    def post_patch_apply(req: PatchApplyRequest) -> dict:
        """Atomically apply patches. ``require_approval=true`` by default."""
        from .patches import apply_patches
        result = apply_patches(req.patches, req.files,
                               require_approval=req.require_approval)
        return result.to_dict()

    @app.get("/memory", dependencies=[_Depends(require_auth)])
    def get_memory(project_root: Optional[str] = None,
                   project_id: Optional[str] = None,
                   scope: str = "all", type: Optional[str] = None,
                   limit: int = 200) -> dict:
        """List memory records. Scope by ``project_id`` (extension /
        server clients) or ``project_root`` (CLI-style path)."""
        from . import memory as mem_mod
        pr = Path(project_root) if project_root else None
        mem = mem_mod.open_memory(project_root=pr, project_id=project_id)
        if scope == "all":
            records = mem.all_records()
        else:
            records = mem.all_records(scope=scope)  # type: ignore[arg-type]
        if type:
            records = [r for r in records if r.type == type]
        from dataclasses import asdict
        return {
            "records": [asdict(r) for r in records[:limit]],
            "total": len(records),
        }

    @app.post("/memory/forget", dependencies=[_Depends(require_auth)])
    def post_memory_forget(req: MemoryForgetRequest) -> dict:
        from . import memory as mem_mod
        pr = Path(req.project_root) if req.project_root else None
        mem = mem_mod.open_memory(project_root=pr, project_id=req.project_id)
        ok = mem.forget(req.record_id, scope=req.scope)  # type: ignore[arg-type]
        return {"ok": ok, "record_id": req.record_id}

    @app.post("/memory/remember", dependencies=[_Depends(require_auth)])
    def post_memory_remember(req: MemoryRememberRequest) -> dict:
        """Write a memory record. Used by the extension's Ignore /
        Mark-accepted actions — user-driven decisions that should
        persist and influence future audits."""
        from . import memory as mem_mod
        pr = Path(req.project_root) if req.project_root else None
        mem = mem_mod.open_memory(project_root=pr, project_id=req.project_id)
        rec = mem.remember(
            type_=req.type,  # type: ignore[arg-type]
            claim_text=req.claim_text,
            paper_key=req.paper_key,
            cite_key=req.cite_key,
            decision=req.decision,
            tier=req.tier,
            confidence=req.confidence,
            source=req.source,
            rationale=req.rationale,
            scope=req.scope,  # type: ignore[arg-type]
        )
        from dataclasses import asdict
        return {"ok": rec is not None, "record": asdict(rec) if rec else None}

    @app.delete("/memory/project", dependencies=[_Depends(require_auth)])
    def delete_memory_project(
        project_root: Optional[str] = None, project_id: Optional[str] = None,
    ) -> dict:
        from . import memory as mem_mod
        pr = Path(project_root) if project_root else None
        mem = mem_mod.open_memory(project_root=pr, project_id=project_id)
        return {"ok": mem.purge_project(), "project_id": mem.project_id}

    @app.get("/cache/status", dependencies=[_Depends(require_auth)])
    def get_cache_status() -> dict:
        """Disk-size summary of every cache directory."""
        from platformdirs import user_cache_dir
        cache = Path(user_cache_dir("bibsync", "bibsync"))
        out: dict = {"cache_root": str(cache), "subdirs": {}}
        for sub in ("paper_content", "pdfs", "embeddings", "memory"):
            p = cache / sub
            if not p.exists():
                out["subdirs"][sub] = {"exists": False, "size_bytes": 0, "n_files": 0}
                continue
            n = 0
            size = 0
            for f in p.rglob("*"):
                if f.is_file():
                    n += 1
                    try:
                        size += f.stat().st_size
                    except OSError:
                        pass
            out["subdirs"][sub] = {"exists": True, "size_bytes": size, "n_files": n}
        return out

    @app.post("/cache/clear", dependencies=[_Depends(require_auth)])
    def post_cache_clear(target: str = "paper_content") -> dict:
        """Clear an optional cache. ``memory/`` is NEVER auto-cleared by this
        endpoint — use the explicit memory CRUD for that."""
        if target == "memory":
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail="memory is not auto-clearable here; use DELETE /memory/project",
            )
        from platformdirs import user_cache_dir
        cache = Path(user_cache_dir("bibsync", "bibsync")) / target
        if not cache.exists():
            return {"ok": True, "cleared": 0}
        n = 0
        for f in cache.rglob("*"):
            if f.is_file():
                try:
                    f.unlink()
                    n += 1
                except OSError:
                    pass
        return {"ok": True, "cleared": n, "target": target}

    @app.get("/privacy", dependencies=[_Depends(require_auth)])
    def get_privacy(
        project_root: Optional[str] = None, project_id: Optional[str] = None,
    ) -> dict:
        """What persistent data does BibSync hold for this project?"""
        from . import memory as mem_mod
        pr = Path(project_root) if project_root else None
        mem = mem_mod.open_memory(project_root=pr, project_id=project_id)
        has_project = pr is not None or project_id is not None
        project_records = mem.all_records(scope="project") if has_project else []
        user_records = mem.all_records(scope="user")
        return {
            "project_root": str(pr) if pr else None,
            "project_id": mem.project_id,
            "memory": {
                "project_records": len(project_records),
                "user_records": len(user_records),
                "user_jsonl_path": str(mem.user_file),
                "project_jsonl_path": (
                    str(mem.project_file) if mem.project_file else None
                ),
            },
            "caches_root": str(get_cache_status()["cache_root"]),
        }

    return app


# ── uvicorn launcher (called from CLI) ──────────────────────────────────────


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 38476,
    log_level: str = "info",
    use_token: bool = False,
) -> None:
    """Launch the server. Blocks until killed.

    Auth model:
      • ``use_token=False`` (default) — no auth. The server is
        127.0.0.1-bound; the network layer is the boundary. Zero-friction
        for the Chrome extension, which then needs no token.
      • ``use_token=True`` — generate a bearer token, write it to
        ``server.token``. Every request must carry it. Use on shared
        machines; the extension's Settings tab has a token field.

    Refuses external binds unless ``BIBSYNC_ALLOW_EXTERNAL=1`` is set —
    manuscript content stays local.
    """
    import uvicorn

    if host not in ("127.0.0.1", "localhost") and not os.environ.get(
        "BIBSYNC_ALLOW_EXTERNAL"
    ):
        raise RuntimeError(
            f"Refusing to bind to {host}. Set BIBSYNC_ALLOW_EXTERNAL=1 to "
            "override (NOT recommended — manuscript content stays local)."
        )

    token = generate_token() if use_token else None
    print(f"[bibsync serve] Listening on http://{host}:{port}")
    if token:
        print(f"[bibsync serve] Bearer token written to {_token_path()}")
        print("[bibsync serve] Paste it into the extension's Settings tab.")
    else:
        print("[bibsync serve] No auth (127.0.0.1-only). Pass --token to require a bearer token.")

    app = create_app(token=token)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
