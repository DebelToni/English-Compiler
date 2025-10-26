MODEL = "granite4:tiny-h"
DEFAULT_MODEL_SUGGESTION = "granite:tiny-h"
"""Utility script to create a file and watch it for save events."""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable


DEFAULT_VENV_PATH = "/tmp/english_compiler_venv"
BASE_TMP_DIR = pathlib.Path("/tmp/english_compiler")
TRACKING_FILE = BASE_TMP_DIR / "tracked.yml"
MODEL_FILES = ("English.py", "Compile.py")


def update_model_constant(new_model: str) -> None:
    """Rewrite the MODEL constant in both scripts."""
    root = pathlib.Path(__file__).resolve().parent
    for name in MODEL_FILES:
        path = root / name
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        lines = content.splitlines()
        if lines:
            if lines[0].startswith("MODEL ="):
                lines[0] = f'MODEL = "{new_model}"'
            else:
                lines.insert(0, f'MODEL = "{new_model}"')
        else:
            lines = [f'MODEL = "{new_model}"']
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def handle_missing_model() -> bool:
    """Prompt for a replacement model and update both scripts."""
    prompt = (
        f"Model {MODEL} is not present, which model to use "
        f"(default {DEFAULT_MODEL_SUGGESTION}): "
    )
    while True:
        try:
            new_model = input(prompt)
        except EOFError:
            return False
        new_model = new_model.strip() or DEFAULT_MODEL_SUGGESTION
        if not new_model:
            continue
        update_model_constant(new_model)
        global MODEL
        MODEL = new_model
        return True


def is_missing_model_message(message: str) -> bool:
    """Determine if an Ollama error message indicates a missing model."""
    lowered = message.lower()
    return "model" in lowered and ("not found" in lowered or "no such model" in lowered)


def notify_endpoint_unavailable() -> None:
    """Emit a standard notice when Ollama cannot be reached."""
    print("Ollama endpoint is not reachable. Is Ollama installed?", file=sys.stderr)


def get_tracking_entry(file_path: str) -> dict[str, Any] | None:
    """Return the tracked metadata for a given file if present."""
    return load_tracking_map().get(file_path)


def query_required_libraries(source_text: str, allow_retry: bool = True) -> list[str]:
    """Ask the local Ollama endpoint which libraries should be installed."""
    payload = {
        "model": MODEL,
        "prompt": (
            "You are an assistant that inspects text files destined to be translated "
            "to Python. Analyze the provided file contents and determine which Python "
            "libraries must be installed (pip-installable names) so the translated "
            "program can execute. Respond using the provided JSON schema.\n\n"
            "File contents:\n"
            f"{source_text}"
        ),
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "libraries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of pip-installable package names.",
                }
            },
            "required": ["libraries"],
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_data = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        try:
            message = error.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            message = str(error)
        if allow_retry and is_missing_model_message(message):
            if handle_missing_model():
                return query_required_libraries(source_text, allow_retry=False)
        print(message, file=sys.stderr)
        return []
    except urllib.error.URLError:
        notify_endpoint_unavailable()
        return []
    except TimeoutError:
        notify_endpoint_unavailable()
        return []

    try:
        outer = json.loads(response_data)
    except json.JSONDecodeError:
        return []

    if isinstance(outer, dict) and "error" in outer:
        message = str(outer["error"])
        if allow_retry and is_missing_model_message(message):
            if handle_missing_model():
                return query_required_libraries(source_text, allow_retry=False)
        print(message, file=sys.stderr)
        return []

    try:
        formatted = outer.get("response", "") if isinstance(outer, dict) else ""
        parsed = json.loads(formatted)
    except (json.JSONDecodeError, TypeError):
        return []

    libraries = parsed.get("libraries")
    if not isinstance(libraries, list):
        return []

    cleaned: list[str] = []
    seen = set()
    for item in libraries:
        if not isinstance(item, str):
            continue
        package = item.strip()
        if package and package not in seen:
            seen.add(package)
            cleaned.append(package)
    return cleaned


def install_libraries(venv_path: str, libraries: list[str]) -> bool:
    """Install the requested libraries inside the tracked virtual environment."""
    if not libraries:
        return True
    python_bin = pathlib.Path(venv_path).expanduser().resolve() / "bin" / "python"
    if not python_bin.exists():
        return False
    try:
        subprocess.run(
            [str(python_bin), "-m", "pip", "install", *libraries],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return False
    return True


def update_requirements(file_path: str) -> None:
    """Trigger the dependency discovery workflow on file save."""
    entry = get_tracking_entry(file_path)
    if not entry:
        return
    venv_path = entry.get("venv")
    if not venv_path:
        return
    try:
        source_text = pathlib.Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return
    libraries = query_required_libraries(source_text)
    existing = set(entry.get("libraries", []))
    recommended = set(libraries)
    if not recommended and not existing:
        return
    to_install = sorted(recommended - existing)
    install_success = True
    if to_install:
        install_success = install_libraries(venv_path, to_install)
    if install_success:
        combined = sorted(existing | recommended)
        record_tracking_entry(file_path, venv_path, combined)


def create_file(target_path: str) -> None:
    """Create the target file, ensuring its parent directory exists."""
    directory = os.path.dirname(target_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(target_path, "a", encoding="utf-8"):
        # Touch the file without modifying its contents.
        pass


def load_tracking_map() -> dict[str, dict[str, Any]]:
    """Read tracked files data from storage."""
    if not TRACKING_FILE.exists():
        return {}
    try:
        content = TRACKING_FILE.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not content.strip():
        return {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback to legacy flat format.
        mapping: dict[str, dict[str, Any]] = {}
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line == "tracked:":
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            mapping[key.strip()] = {
                "venv": value.strip(),
                "libraries": [],
                "cache": {},
            }
        return mapping
    tracked = data.get("tracked", {})
    mapping: dict[str, dict[str, Any]] = {}
    if isinstance(tracked, dict):
        for key, value in tracked.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, dict):
                venv_path = value.get("venv", "")
                libraries = value.get("libraries", [])
                cache_value = value.get("cache", {})
            elif isinstance(value, str):
                venv_path = value
                libraries = []
                cache_value = {}
            else:
                continue
            lib_list = [
                lib for lib in libraries if isinstance(lib, str) and lib.strip()
            ]
            cache_dict = cache_value if isinstance(cache_value, dict) else {}
            mapping[key] = {
                "venv": venv_path,
                "libraries": lib_list,
                "cache": cache_dict,
            }
    return mapping


def save_tracking_map(mapping: dict[str, dict[str, Any]]) -> None:
    """Persist tracked files mapping in the tmp directory."""
    BASE_TMP_DIR.mkdir(parents=True, exist_ok=True)
    payload_tracked: dict[str, Any] = {}
    for key, value in mapping.items():
        entry_payload: dict[str, Any] = {
            "venv": value.get("venv", ""),
            "libraries": value.get("libraries", []),
        }
        cache_value = value.get("cache")
        if isinstance(cache_value, dict) and cache_value:
            entry_payload["cache"] = cache_value
        payload_tracked[key] = entry_payload
    payload = {"tracked": payload_tracked}
    TRACKING_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_tracking_entry(
    file_path: str,
    venv_path: str,
    libraries: list[str] | None = None,
) -> None:
    """Persist the mapping between a file and its virtual environment."""
    tracking = load_tracking_map()
    entry = tracking.get(file_path)
    if entry is None:
        entry = {"venv": venv_path, "libraries": [], "cache": {}}
    entry["venv"] = venv_path
    if libraries is not None:
        entry["libraries"] = libraries
        entry["cache"] = {}
    else:
        entry.setdefault("cache", {})
    tracking[file_path] = entry
    save_tracking_map(tracking)


def watch_file_for_saves(
    file_path: str,
    callback: Callable[[str], None],
    poll_interval: float = 0.5,
) -> None:
    """Continuously watch `file_path` and invoke `callback` on save events."""
    last_modified = None
    while True:
        try:
            modified_time = os.path.getmtime(file_path)
        except FileNotFoundError:
            # Reset if the file is temporarily unavailable.
            last_modified = None
        else:
            if last_modified is None:
                last_modified = modified_time
            elif modified_time != last_modified:
                last_modified = modified_time
                callback(file_path)
        time.sleep(poll_interval)


def watch_worker(file_path: str) -> None:
    """Entry point for the detached watcher subprocess."""
    watch_file_for_saves(file_path, update_requirements)


def launch_watcher(file_path: str) -> None:
    """Launch the watcher in a detached subprocess."""
    subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--watch", file_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a file and watch it for save events.",
    )
    parser.add_argument(
        "filename",
        help="Path to the file that should be created and watched.",
    )
    parser.add_argument(
        "--venv",
        dest="venv_path",
        default=os.environ.get("ENGLISH_COMPILER_VENV", DEFAULT_VENV_PATH),
        help=(
            "Path to the Python virtual environment whose interpreter should be shared. "
            "Defaults to $ENGLISH_COMPILER_VENV or /tmp/english_compiler_venv."
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    target_path = os.path.abspath(args.filename)

    if args.watch:
        watch_worker(target_path)
        return

    venv_path = os.path.abspath(args.venv_path)

    create_file(target_path)
    record_tracking_entry(target_path, venv_path)

    launch_watcher(target_path)


if __name__ == "__main__":
    main()
