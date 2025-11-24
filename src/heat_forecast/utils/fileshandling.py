from __future__ import annotations
import inspect
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Optional

def _looks_dangerous(path: Path) -> bool:
    # refuse to remove root, drive roots, home, or extremely short paths
    p = path.resolve()
    if str(p) in ("/", "\\") or p == p.anchor:  # UNIX root or Windows drive root
        return True
    home = Path.home().resolve()
    if p == home or home in p.parents:
        # allow removing inside home, but refuse *exactly* home itself
        return p == home
    # guard against very short likely-mistyped dirs (e.g., "/tmp", "C:\\")
    return len(p.parts) <= 2

def _chmod_writable(p: str | os.PathLike) -> None:
    try:
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    except Exception:
        pass

def _onerror(func, path, exc_info):
    _, exc, _ = exc_info
    if isinstance(exc, PermissionError):
        _chmod_writable(path)
        try:
            func(path)
            return
        except Exception:
            pass
    if isinstance(exc, FileNotFoundError):
        return
    raise

def _onexc(path, exc):
    if isinstance(exc, PermissionError):
        _chmod_writable(path)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                os.rmdir(path)
            else:
                os.remove(path)
            return
        except Exception:
            pass
    if isinstance(exc, FileNotFoundError):
        return
    raise exc

def _supports_onexc() -> bool:
    # Detect if API supports onexc 
    params = inspect.signature(shutil.rmtree).parameters
    return "onexc" in params

def remove_tree(
    path: os.PathLike | str,
    *,
    retries: int = 3,
    delay: float = 0.15,
    require_within: Optional[Path] = None,
) -> None:
    """
    Robust rmtree:
      - Works on Python 3.10+ (onerror / onexc auto-detected)
      - Retries transient failures
      - Optional safety fence via `require_within` (e.g., BASE_DIR)
    """
    p = Path(path)
    if not p.exists():
        return

    if _looks_dangerous(p):
        raise ValueError(f"Refusing to remove suspicious path: {p!s}")
    if require_within is not None:
        base = Path(require_within).resolve()
        if base not in p.resolve().parents:
            raise ValueError(f"{p!s} is not inside {base!s}")

    handler_kw = {"onexc": _onexc} if _supports_onexc() else {"onerror": _onerror}

    last_exc = None
    for attempt in range(retries + 1):
        try:
            shutil.rmtree(p, **handler_kw)
            return
        except FileNotFoundError:
            return
        except PermissionError as e:
            last_exc = e
        except OSError as e:
            # Windows sharing violations / transient 'in use'
            last_exc = e
        if attempt < retries:
            time.sleep(delay)
    # If we got here, all retries failed
    if last_exc is not None:
        raise last_exc
