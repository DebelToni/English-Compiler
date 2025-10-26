MODEL = "granite4:tiny-h"
DEFAULT_MODEL_SUGGESTION = "granite:tiny-h"
"""Compile script that requests Python code generation and executes it."""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from typing import Any, List, Tuple

DEFAULT_VENV_PATH = "/tmp/english_compiler_venv"
BASE_TMP_DIR = pathlib.Path("/tmp/english_compiler")
TRACKING_FILE = BASE_TMP_DIR / "tracked.yml"
CACHE_DIR = BASE_TMP_DIR / "cache"
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


def load_tracking_map() -> dict[str, dict[str, Any]]:
    """Read tracked files data stored by English.py."""
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


def get_tracking_entry(file_path: pathlib.Path) -> dict[str, Any] | None:
    """Return metadata for the requested file."""
    return load_tracking_map().get(str(file_path))


def resolve_environment(source_file: pathlib.Path) -> Tuple[str, list[str]]:
    """Locate the interpreter and libraries from the tracked environment."""
    entry = get_tracking_entry(source_file)
    env_override = os.environ.get("ENGLISH_COMPILER_VENV")
    venv_root = env_override or (entry.get("venv") if entry else None) or DEFAULT_VENV_PATH
    venv_python = pathlib.Path(venv_root).expanduser().resolve() / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(
            f"Virtual environment Python not found at {venv_python}. "
            "Ensure the file is tracked by English.py or set ENGLISH_COMPILER_VENV."
        )
    libraries = entry.get("libraries", []) if entry else []
    return str(venv_python), libraries


def build_history_summary(history: List[dict[str, Any]]) -> str:
    """Summarize previous attempts for the LLM prompt."""
    if not history:
        return "No previous attempts."
    lines: List[str] = []
    for index, attempt in enumerate(history, start=1):
        returncode = attempt.get("returncode")
        stdout = (attempt.get("stdout", "") or "").strip()
        stderr = (attempt.get("stderr", "") or "").strip()
        snippet = (attempt.get("code", "") or "").strip()
        snippet = snippet[:2000]
        stdout = stdout[:2000]
        stderr = stderr[:2000]
        lines.append(f"Attempt {index}: returncode={returncode}")
        if snippet:
            lines.append("Code snippet:\n" + snippet)
        if stdout:
            lines.append("Stdout:\n" + stdout)
        if stderr:
            lines.append("Stderr:\n" + stderr)
    return "\n\n".join(lines)


def request_python_code(
    source_text: str,
    libraries: list[str],
    history: List[dict[str, Any]],
) -> str:
    return _request_python_code(source_text, libraries, history, allow_retry=True)


def _request_python_code(
    source_text: str,
    libraries: list[str],
    history: List[dict[str, Any]],
    allow_retry: bool,
) -> str:
    """Ask the local Ollama endpoint to translate the text into Python."""
    libs_text = ", ".join(libraries) if libraries else "no additional libraries"
    history_summary = build_history_summary(history)
    payload = {
        "model": MODEL,
        "prompt": (
            "You translate text specifications into executable Python scripts. "
            "Only return a Python program that fulfills the specification. "
            "Use the following pip-installed libraries when helpful:\n"
            f"{libs_text}\n\n"
            "Specification:\n"
            f"{source_text}\n\n"
            "Previous attempts and their results:\n"
            f"{history_summary}\n\n"
            "Improve on prior failures and output only the Python code."
        ),
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python source code implementing the specification.",
                }
            },
            "required": ["code"],
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
        with urllib.request.urlopen(request, timeout=120) as response:
            response_data = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        try:
            message = error.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            message = str(error)
        if allow_retry and is_missing_model_message(message):
            if handle_missing_model():
                return _request_python_code(
                    source_text,
                    libraries,
                    history,
                    allow_retry=False,
                )
        print(message, file=sys.stderr)
        return ""
    except urllib.error.URLError:
        notify_endpoint_unavailable()
        return ""
    except TimeoutError:
        notify_endpoint_unavailable()
        return ""

    try:
        outer = json.loads(response_data)
    except json.JSONDecodeError:
        return ""

    if isinstance(outer, dict) and "error" in outer:
        message = str(outer["error"])
        if allow_retry and is_missing_model_message(message):
            if handle_missing_model():
                return _request_python_code(
                    source_text,
                    libraries,
                    history,
                    allow_retry=False,
                )
        print(message, file=sys.stderr)
        return ""

    try:
        formatted = outer.get("response", "") if isinstance(outer, dict) else ""
        parsed = json.loads(formatted)
    except (json.JSONDecodeError, TypeError):
        return ""

    code = parsed.get("code")
    return code if isinstance(code, str) else ""


def evaluate_execution(
    specification: str,
    stdout: str,
    stderr: str,
    returncode: int,
) -> bool:
    return _evaluate_execution(specification, stdout, stderr, returncode, allow_retry=True)


def _evaluate_execution(
    specification: str,
    stdout: str,
    stderr: str,
    returncode: int,
    allow_retry: bool,
) -> bool:
    """Ask the LLM if the execution result satisfies the specification."""
    payload = {
        "model": MODEL,
        "prompt": (
            "You evaluate whether program output satisfies a specification. "
            "Respond using the JSON schema and answer only 'Yes' or 'No'. "
            "If errors occur or the result does not fulfill the spec, respond 'No'.\n\n"
            f"Specification:\n{specification}\n\n"
            f"Return code: {returncode}\n"
            f"Stdout:\n{stdout or '[empty]'}\n\n"
            f"Stderr:\n{stderr or '[empty]'}\n"
        ),
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": ["Yes", "No"],
                    "description": "Whether the program output satisfies the specification.",
                }
            },
            "required": ["answer"],
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
        with urllib.request.urlopen(request, timeout=120) as response:
            response_data = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        try:
            message = error.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            message = str(error)
        if allow_retry and is_missing_model_message(message):
            if handle_missing_model():
                return _evaluate_execution(
                    specification,
                    stdout,
                    stderr,
                    returncode,
                    allow_retry=False,
                )
        print(message, file=sys.stderr)
        return False
    except urllib.error.URLError:
        notify_endpoint_unavailable()
        return False
    except TimeoutError:
        notify_endpoint_unavailable()
        return False

    try:
        outer = json.loads(response_data)
    except json.JSONDecodeError:
        return False

    if isinstance(outer, dict) and "error" in outer:
        message = str(outer["error"])
        if allow_retry and is_missing_model_message(message):
            if handle_missing_model():
                return _evaluate_execution(
                    specification,
                    stdout,
                    stderr,
                    returncode,
                    allow_retry=False,
                )
        print(message, file=sys.stderr)
        return False

    try:
        formatted = outer.get("response", "") if isinstance(outer, dict) else ""
        parsed = json.loads(formatted)
    except (json.JSONDecodeError, TypeError):
        return False

    answer = parsed.get("answer")
    return isinstance(answer, str) and answer.strip() == "Yes"


def persist_cache_code(spec_hash: str, code: str) -> pathlib.Path:
    """Store generated code for reuse."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{spec_hash}.py"
    cache_file.write_text(code, encoding="utf-8")
    return cache_file


def update_cache_metadata(source_file: pathlib.Path, spec_hash: str, code_path: pathlib.Path) -> None:
    """Persist cache metadata for the source file."""
    mapping = load_tracking_map()
    key = str(source_file)
    entry = mapping.get(key, {"venv": DEFAULT_VENV_PATH, "libraries": [], "cache": {}})
    entry["cache"] = {
        "spec_hash": spec_hash,
        "code_path": str(code_path),
    }
    mapping[key] = entry
    save_tracking_map(mapping)


def clear_cache_metadata(source_file: pathlib.Path) -> None:
    """Remove cached code metadata (and file) on failure."""
    mapping = load_tracking_map()
    key = str(source_file)
    entry = mapping.get(key)
    if not entry:
        return
    cache_info = entry.pop("cache", None)
    mapping[key] = entry
    save_tracking_map(mapping)
    if isinstance(cache_info, dict):
        code_path = cache_info.get("code_path")
        if isinstance(code_path, str):
            cache_file = pathlib.Path(code_path)
            try:
                cache_file.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def try_run_cached(
    source_file: pathlib.Path,
    venv_python: str,
    spec_hash: str,
) -> Tuple[bool, List[dict[str, Any]]]:
    """Attempt to run cached code without hitting the LLM."""
    entry = get_tracking_entry(source_file)
    if not entry:
        return False, []
    cache_info = entry.get("cache")
    if not isinstance(cache_info, dict):
        return False, []
    if cache_info.get("spec_hash") != spec_hash:
        return False, []
    code_path = cache_info.get("code_path")
    if not isinstance(code_path, str):
        return False, []
    code_file = pathlib.Path(code_path)
    if not code_file.exists():
        return False, []
    try:
        cached_code = code_file.read_text(encoding="utf-8")
    except OSError:
        cached_code = ""

    result = subprocess.run(
        [venv_python, str(code_file)],
        capture_output=True,
        text=True,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode == 0:
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        return True, []

    clear_cache_metadata(source_file)
    history_entry = {
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "code": cached_code,
    }
    return False, [history_entry]


def generate_code(source_text: str, source_file: pathlib.Path) -> None:
    """Request Python code from the LLM and execute it."""
    venv_python, libraries = resolve_environment(source_file)
    spec_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()

    cache_used, cached_history = try_run_cached(source_file, venv_python, spec_hash)
    if cache_used:
        return

    history: List[dict[str, Any]] = list(cached_history)
    MAX_ATTEMPTS = 5

    for attempt in range(1, MAX_ATTEMPTS + 1):
        code = request_python_code(source_text, libraries, history)
        if not code.strip():
            print("Failed to obtain Python code from the model.", file=sys.stderr)
            raise SystemExit(1)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir="/tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp_file:
            tmp_file.write(code)
            tmp_path = tmp_file.name

        try:
            result = subprocess.run(
                [venv_python, tmp_path],
                capture_output=True,
                text=True,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            returncode = result.returncode
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

        success = returncode == 0 and evaluate_execution(source_text, stdout, stderr, returncode)
        if success:
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            cache_file = persist_cache_code(spec_hash, code)
            update_cache_metadata(source_file, spec_hash, cache_file)
            return

        history.append(
            {
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "code": code,
            }
        )

    print("Compilation loop failed to satisfy the specification.", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a file, request Python code, and execute it inside the tracked venv.",
    )
    parser.add_argument(
        "filename",
        help="Path to the source file to compile.",
    )
    args = parser.parse_args()

    target_path = pathlib.Path(args.filename).expanduser().resolve()
    source_text = target_path.read_text(encoding="utf-8")
    generate_code(source_text, target_path)


if __name__ == "__main__":
    main()
