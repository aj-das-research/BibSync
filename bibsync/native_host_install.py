"""Install the Chrome native-messaging host manifest.

Chrome looks for native hosts in per-OS well-known directories:

  macOS:    ~/Library/Application Support/Google/Chrome/NativeMessagingHosts/com.bibsync.host.json
            (Chromium: ~/Library/Application Support/Chromium/NativeMessagingHosts/)
  Linux:    ~/.config/google-chrome/NativeMessagingHosts/com.bibsync.host.json
            (Chromium: ~/.config/chromium/NativeMessagingHosts/)
  Windows:  HKCU\\Software\\Google\\Chrome\\NativeMessagingHosts\\com.bibsync.host

Each manifest lists:
  • the absolute path of the Python script Chrome should launch
  • the extension ID(s) allowed to talk to it

For development/personal use we write the manifest to the user-scope
directory and pin to a single extension ID supplied by the user (or
"*" for unsafe-but-easy local dev).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

HOST_NAME = "com.bibsync.host"


def native_host_script_path() -> Path:
    """Locate ``bibsync_native_host.py`` relative to the installed package.

    Walks up from the package directory looking for ``native-host/``.
    In a dev install (editable / from a repo checkout) this resolves to
    ``<repo>/native-host/bibsync_native_host.py``. In a future pip-installed
    distribution we'd vendor the script into the package; for now the
    dev path is the only one users have."""
    here = Path(__file__).resolve()
    # bibsync/native_host_install.py → bibsync/ → repo-root → native-host/
    repo_root = here.parent.parent
    candidate = repo_root / "native-host" / "bibsync_native_host.py"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"native-host/bibsync_native_host.py not found at {candidate}. "
        "Either you're running a non-dev install or the repo layout changed."
    )


def chrome_manifest_dir() -> Path:
    """Per-OS Chrome (stable channel) native-messaging directory."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "NativeMessagingHosts"
    if sys.platform.startswith("linux"):
        return Path.home() / ".config" / "google-chrome" / "NativeMessagingHosts"
    if sys.platform == "win32":
        # Windows uses the registry, not files. We write a file too for
        # parity but Chrome won't find it without a registry entry.
        return Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "NativeMessagingHosts"
    raise RuntimeError(f"unsupported platform: {sys.platform}")


def manifest_path() -> Path:
    return chrome_manifest_dir() / f"{HOST_NAME}.json"


def build_manifest(*, extension_id: str, python_executable: Optional[str] = None) -> dict:
    """Construct the JSON manifest Chrome expects.

    ``extension_id`` is the 32-char Chrome extension ID (visible at
    chrome://extensions when developer mode is on). Use "*" only for
    local dev — it allows ANY extension to launch the host.
    """
    script = native_host_script_path()
    if python_executable is None:
        # Use the current interpreter — same one running ``bibsync``.
        # Encoded as an absolute path so Chrome can find it.
        python_executable = sys.executable

    # Chrome's manifest "path" field has to be a single executable; we use
    # a small wrapper script (or `python /path/to/host.py`) but the
    # manifest can't have arguments. Instead, write a launcher script
    # that runs python with the host script.
    # On non-Windows, the simplest is to make the host script itself
    # executable + use a shebang. The script is already chmod +x'd by the
    # commit, but if the shebang interpreter isn't a python3 with the
    # standard library, the host will fail.
    #
    # Robust approach: write a tiny shell wrapper into the same dir as
    # the manifest, and point at THAT. Wrapper invokes the project's
    # python explicitly.

    if sys.platform == "win32":
        wrapper_name = f"{HOST_NAME}_launcher.bat"
    else:
        wrapper_name = f"{HOST_NAME}_launcher.sh"

    wrapper_path = chrome_manifest_dir() / wrapper_name

    extension_origin = (
        f"chrome-extension://{extension_id}/"
        if extension_id != "*"
        else "chrome-extension://*/"
    )

    return {
        "name": HOST_NAME,
        "description": "BibSync citation AI bridge (Chrome → bibsync serve)",
        "path": str(wrapper_path),
        "type": "stdio",
        "allowed_origins": [extension_origin],
    }


def _write_wrapper_script(python_exe: str) -> Path:
    """Write a launcher script that Chrome calls — invokes python with the
    host. A wrapper is needed because Chrome's manifest can't pass args
    to the executable."""
    host = native_host_script_path()
    if sys.platform == "win32":
        wrapper_path = chrome_manifest_dir() / f"{HOST_NAME}_launcher.bat"
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        wrapper_path.write_text(
            f'@echo off\r\n"{python_exe}" "{host}" %*\r\n',
            encoding="utf-8",
        )
    else:
        wrapper_path = chrome_manifest_dir() / f"{HOST_NAME}_launcher.sh"
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        wrapper_path.write_text(
            f"#!/bin/sh\nexec {python_exe!r} {str(host)!r} \"$@\"\n",
            encoding="utf-8",
        )
        try:
            os.chmod(wrapper_path, 0o755)
        except OSError:
            pass
    return wrapper_path


def install(
    *, extension_id: str = "*", python_executable: Optional[str] = None,
) -> Path:
    """Install the native-messaging manifest for the current user.

    Returns the manifest path written. Does NOT touch the Windows registry
    yet — that's a Sprint-F task; Windows users currently need to add the
    registry entry manually (instructions in README)."""
    py = python_executable or sys.executable
    chrome_manifest_dir().mkdir(parents=True, exist_ok=True)
    _write_wrapper_script(py)
    manifest = build_manifest(extension_id=extension_id, python_executable=py)
    p = manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return p


def uninstall() -> bool:
    """Remove the manifest + wrapper script. Returns True if anything
    was deleted."""
    removed = False
    p = manifest_path()
    if p.exists():
        p.unlink()
        removed = True
    for wrapper_name in (f"{HOST_NAME}_launcher.sh", f"{HOST_NAME}_launcher.bat"):
        w = chrome_manifest_dir() / wrapper_name
        if w.exists():
            w.unlink()
            removed = True
    return removed


def status() -> dict:
    """Inspect the current install state — what's on disk."""
    p = manifest_path()
    info: dict = {
        "manifest_path": str(p),
        "installed": p.exists(),
        "host_script": str(native_host_script_path()),
        "wrapper": None,
        "allowed_origins": None,
    }
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            info["wrapper"] = data.get("path")
            info["allowed_origins"] = data.get("allowed_origins")
        except (OSError, json.JSONDecodeError):
            pass
    return info
