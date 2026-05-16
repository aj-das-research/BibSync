"""Patch model — every text edit the AI proposes flows through this.

Two layers:

  • **Mechanics**: a Patch is a (file, range, new_text) edit. Apply is
    atomic over a set of patches: either ALL succeed or NONE do. Shape
    is LSP-compatible (TextEdit-like) so a future VS Code adapter
    works trivially.

  • **Semantics**: ``type`` field carries the *intent* (insert_citation,
    replace_citation, append_bibtex, …) so the UI can render
    appropriate confirmations and the user can refuse a class of
    changes without inspecting every diff.

Hard contract:

  1. The extension NEVER edits the manuscript directly. It builds a
     list of Patch objects, sends them to ``/patch/preview`` for a
     diff render, and only sends ``/patch/apply`` after the user
     approves.

  2. ``apply()`` re-verifies that ``old_text`` matches the current
     file content at the given range before applying. If it doesn't,
     the patch becomes a *conflict* — the file changed since the
     patch was constructed. None of the conflicting batch applies.

  3. Patches are SELF-CONTAINED: they carry their own range/old_text/
     new_text so they round-trip through JSON without needing
     reference to the original audit run.
"""

from __future__ import annotations

import difflib
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional


# The semantic types currently supported by the patch applier. The text
# mechanic for each is the same (replace range with new_text); the type
# exists for UI rendering + audit-log clarity.
PATCH_TYPES = frozenset({
    "insert_citation",       # add \cite{key} at offset
    "replace_citation",      # swap one key for another (or for a marker)
    "remove_citation",       # delete one \cite{} call (with comment marker)
    "append_bibtex",         # add an entry to the end of a .bib file
    "replace_bibtex_entry",  # swap an entire entry in a .bib file
    "rename_cite_key",       # rename across .tex + .bib (multi-patch)
    "add_comment",           # inline %-comment with reasoning
    "raw",                   # escape hatch for tests + future use
})


@dataclass
class Range:
    """Half-open character range [start, end) within a file."""
    start: int
    end: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.start, self.end)


@dataclass
class Patch:
    """One text edit to one file. Always carries its own old_text so the
    applier can detect conflicts (file changed since patch was built)."""

    patch_id: str
    type: str
    file: str
    range: Range
    old_text: str
    new_text: str
    reason: str = ""
    issue_id: str = ""           # links back to the audit issue, if any
    user_approved: bool = False  # set true by the client before /patch/apply

    @classmethod
    def new(
        cls,
        *,
        type: str,
        file: str,
        start: int,
        end: int,
        old_text: str,
        new_text: str,
        reason: str = "",
        issue_id: str = "",
    ) -> "Patch":
        """Construct a Patch with a fresh id. The patch_id is opaque to
        callers — used only for forget/undo references later."""
        if type not in PATCH_TYPES:
            raise ValueError(f"unknown patch type: {type!r} (allowed: {sorted(PATCH_TYPES)})")
        if start < 0 or end < start:
            raise ValueError(f"invalid range: start={start}, end={end}")
        return cls(
            patch_id="patch_" + uuid.uuid4().hex[:10],
            type=type,
            file=file,
            range=Range(start=start, end=end),
            old_text=old_text,
            new_text=new_text,
            reason=reason,
            issue_id=issue_id,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        # Flatten the range nested-dict for friendlier JSON.
        d["range"] = {"start": self.range.start, "end": self.range.end}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Patch":
        r = d.get("range") or {}
        return cls(
            patch_id=d["patch_id"],
            type=d["type"],
            file=d["file"],
            range=Range(start=int(r.get("start", 0)), end=int(r.get("end", 0))),
            old_text=d.get("old_text", ""),
            new_text=d.get("new_text", ""),
            reason=d.get("reason", ""),
            issue_id=d.get("issue_id", ""),
            user_approved=bool(d.get("user_approved", False)),
        )


# ── application ─────────────────────────────────────────────────────────────


@dataclass
class PatchConflict:
    """A patch whose ``old_text`` doesn't match current file content."""
    patch_id: str
    file: str
    expected: str
    actual: str
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PatchResult:
    """Result of an ``apply`` invocation. Atomic — either ``ok=True`` and
    every file in ``files`` was updated, OR ``ok=False`` and the inputs
    are returned unchanged. ``conflicts`` is non-empty only when ok=False."""

    ok: bool
    files: dict       # filename → final content
    conflicts: list   # list[PatchConflict.to_dict()]
    applied: list     # list[patch_id] applied (empty when ok=False)
    errors: list      # human-readable error strings

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "files": self.files,
            "conflicts": [c if isinstance(c, dict) else c.to_dict() for c in self.conflicts],
            "applied": self.applied,
            "errors": self.errors,
        }


def _apply_one(content: str, patch: Patch) -> tuple[Optional[str], Optional[PatchConflict]]:
    """Apply one patch to one file's content. Returns (new_content, None)
    on success or (None, PatchConflict) on conflict."""
    start, end = patch.range.as_tuple()
    if end > len(content):
        return None, PatchConflict(
            patch_id=patch.patch_id, file=patch.file,
            expected=patch.old_text,
            actual=content[start: min(start + 80, len(content))],
            reason=f"range end {end} exceeds file length {len(content)}",
        )
    actual_slice = content[start:end]
    # When old_text is provided, demand exact match. Empty old_text is
    # treated as "insert at offset" (no conflict check).
    if patch.old_text and actual_slice != patch.old_text:
        return None, PatchConflict(
            patch_id=patch.patch_id, file=patch.file,
            expected=patch.old_text,
            actual=actual_slice,
            reason="old_text mismatch — file changed since patch was built",
        )
    return content[:start] + patch.new_text + content[end:], None


def apply_patches(
    patches: list,         # list[Patch] or list[dict]
    files: dict,           # filename → original content
    *,
    require_approval: bool = True,
) -> PatchResult:
    """Atomically apply ``patches`` to ``files``. All-or-nothing.

    When ``require_approval=True`` (the default), patches missing
    ``user_approved=True`` are rejected before any change is made.
    This is the safety contract: extension must explicitly mark a
    patch approved (via the UI accept button) before it can be applied.

    Multi-patch-per-file: patches to the same file are applied in
    REVERSE-OFFSET order so earlier offsets stay valid as later edits
    shift the content. Patches that overlap each other in the same
    file are an error (not a conflict — those are caller bugs).
    """
    # Normalise to Patch dataclass instances.
    normed: list[Patch] = []
    for p in patches:
        if isinstance(p, Patch):
            normed.append(p)
        else:
            normed.append(Patch.from_dict(p))

    if require_approval:
        unapproved = [p for p in normed if not p.user_approved]
        if unapproved:
            return PatchResult(
                ok=False, files=files, conflicts=[], applied=[],
                errors=[
                    f"patch {p.patch_id} is not user-approved — "
                    "set user_approved=true to apply" for p in unapproved
                ],
            )

    # Group patches by file; verify no in-file overlap.
    by_file: dict[str, list[Patch]] = {}
    for p in normed:
        by_file.setdefault(p.file, []).append(p)

    for fname, fpatches in by_file.items():
        # Sort ascending and verify ranges don't overlap.
        sorted_p = sorted(fpatches, key=lambda x: x.range.start)
        for a, b in zip(sorted_p, sorted_p[1:]):
            if a.range.end > b.range.start:
                return PatchResult(
                    ok=False, files=files, conflicts=[], applied=[],
                    errors=[
                        f"overlapping patches in {fname}: "
                        f"{a.patch_id} ({a.range.start}-{a.range.end}) "
                        f"vs {b.patch_id} ({b.range.start}-{b.range.end})"
                    ],
                )

    # Detect conflicts BEFORE applying anything — atomic semantics.
    conflicts: list[PatchConflict] = []
    for fname, fpatches in by_file.items():
        content = files.get(fname)
        if content is None:
            return PatchResult(
                ok=False, files=files, conflicts=[], applied=[],
                errors=[f"patch targets unknown file: {fname!r}"],
            )
        # Probe each patch against the ORIGINAL content (not the
        # in-progress edited content) — overlap was already verified.
        for p in fpatches:
            _, conflict = _apply_one(content, p)
            if conflict is not None:
                conflicts.append(conflict)

    if conflicts:
        return PatchResult(
            ok=False, files=files,
            conflicts=[c.to_dict() for c in conflicts],
            applied=[],
            errors=[],
        )

    # All clear — apply in reverse-offset order per file.
    new_files = dict(files)
    applied: list[str] = []
    for fname, fpatches in by_file.items():
        content = new_files[fname]
        # Reverse-offset so earlier offsets aren't shifted.
        for p in sorted(fpatches, key=lambda x: x.range.start, reverse=True):
            updated, conflict = _apply_one(content, p)
            if conflict is not None:  # belt-and-suspenders — shouldn't happen
                return PatchResult(
                    ok=False, files=files,
                    conflicts=[conflict.to_dict()], applied=[],
                    errors=["unexpected conflict during apply phase"],
                )
            content = updated  # type: ignore[assignment]
            applied.append(p.patch_id)
        new_files[fname] = content

    return PatchResult(ok=True, files=new_files, conflicts=[], applied=applied, errors=[])


def preview_patches(patches: list, files: dict) -> dict:
    """Apply patches WITHOUT requiring approval and WITHOUT mutating the
    inputs; return before/after content + unified diff per file.

    Used by ``/patch/preview`` — the extension renders the diff in a
    modal, the user clicks Accept, the extension flips
    ``user_approved=true`` on each patch and calls ``/patch/apply``.
    """
    # Temporarily mark all patches approved for the preview application.
    normed: list[Patch] = []
    for p in patches:
        pp = p if isinstance(p, Patch) else Patch.from_dict(p)
        normed.append(Patch(
            patch_id=pp.patch_id, type=pp.type, file=pp.file,
            range=pp.range, old_text=pp.old_text, new_text=pp.new_text,
            reason=pp.reason, issue_id=pp.issue_id,
            user_approved=True,  # preview-mode override
        ))
    result = apply_patches(normed, files, require_approval=False)

    out: dict = {"preview": {}, "conflicts": result.conflicts, "ok": result.ok}
    for fname in result.files.keys():
        before = files.get(fname, "")
        after = result.files.get(fname, "")
        if before == after:
            continue
        diff = "\n".join(difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile=f"a/{fname}", tofile=f"b/{fname}", lineterm="",
        ))
        out["preview"][fname] = {
            "before": before,
            "after": after,
            "diff_unified": diff,
        }
    return out


# ── builders — convenience constructors for the common semantic types ──────


def insert_citation_patch(
    *, file: str, offset: int, cite_key: str, reason: str = "", issue_id: str = "",
) -> Patch:
    """``insert \\cite{key}`` at ``offset``. Empty old_text → no conflict
    check against existing content (we're adding, not replacing)."""
    return Patch.new(
        type="insert_citation", file=file, start=offset, end=offset,
        old_text="", new_text=f"\\cite{{{cite_key}}}",
        reason=reason, issue_id=issue_id,
    )


def replace_citation_patch(
    *, file: str, start: int, end: int, old_cite: str, new_cite: str,
    reason: str = "", issue_id: str = "",
) -> Patch:
    """Replace ``\\cite{old}`` with ``\\cite{new}``."""
    return Patch.new(
        type="replace_citation", file=file, start=start, end=end,
        old_text=f"\\cite{{{old_cite}}}", new_text=f"\\cite{{{new_cite}}}",
        reason=reason, issue_id=issue_id,
    )


def append_bibtex_patch(
    *, file: str, current_content: str, entry_text: str,
    reason: str = "", issue_id: str = "",
) -> Patch:
    """Append a BibTeX entry to the end of a .bib file. ``current_content``
    is needed so we can locate the EOF offset."""
    offset = len(current_content)
    # Ensure a leading newline if file doesn't end with one.
    prefix = "" if current_content.endswith("\n") else "\n"
    return Patch.new(
        type="append_bibtex", file=file, start=offset, end=offset,
        old_text="", new_text=prefix + entry_text.rstrip() + "\n",
        reason=reason, issue_id=issue_id,
    )
