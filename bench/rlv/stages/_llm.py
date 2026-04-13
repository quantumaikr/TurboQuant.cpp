"""Shared LLM call utility for all RLV stages.

Talks to a long-running quant-server HTTP process via the OpenAI-style
/v1/chat/completions endpoint. The server is started once per RLV
session (the orchestrator handles startup/teardown via start_server()
and stop_server()) and the model stays resident in memory across all
stage calls. This is the difference between ~5 minutes per question
(subprocess-per-call, model reloaded every time) and ~10 seconds per
question (server, model loaded once).

Enforces the cliff invariant: every prompt must be smaller than the
model's effective working memory (see docs/phase3_rlv_challenge.md §3.2).
"""
import atexit
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
# Day 4: Phi-3.5-mini via quant-server-unified (built on quant.h directly).
# The old libturboquant-based server had a forward-pass sync divergence
# that produced garbage for Phi-3.5/SmolLM2. The unified server compiles
# quant.h as a single translation unit — no sync issues.
# Phi-3.5: ~1.15 tok/s (CPU NEON), ~6.5 tok/s reported in PR #79.
# Q8_0 is 2x faster than Q4_K_M on NEON (simpler dequant, 3.0 vs 1.5 tok/s).
DEFAULT_MODEL = REPO / "models" / "Phi-3.5-mini-instruct-Q8_0.gguf"
DEFAULT_SERVER_BINARY = REPO / "build_metal" / "quant-server-unified"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8421  # arbitrary, avoid conflicts with 8080

# Phase 1B cliff measurements. NEVER set a stage prompt larger than this.
# Phi-3.5-mini has LongRoPE (128K nominal context) and is Q4_K_M with
# turbo_kv_4b compression — its cliff should be at least as large as
# Llama-3.2-3B's (1024 tokens). Conservative estimate: keep 1024 until
# we measure it directly on Phi-3.5.
CLIFF_BUDGET = {
    "models/Llama-3.2-3B-Instruct-Q8_0.gguf": 1024,
    "models/Llama-3.2-1B-Instruct-Q8_0.gguf": 512,
    "models/Phi-3.5-mini-instruct-Q8_0.gguf": 1024,
    "models/Phi-3.5-mini-instruct-Q4_K_M.gguf": 1024,
}


@dataclass
class LLMResult:
    text: str          # the model's generated text (between --- delimiters)
    raw: str           # the full CLI stdout+stderr
    n_tokens: int      # generated token count
    elapsed: float     # wall seconds
    is_error: bool = False  # True if the call failed (text contains error message)


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: 1 token per ~3 chars (wikitext is dense).
    Used by the cliff-budget check, NOT by the actual tokenizer.
    Conservative side: overestimate so we never exceed the cliff."""
    return max(1, len(text) // 3)


def cliff_budget_for(model_path: str) -> int:
    """Return the cliff-safe prompt token budget for the given model."""
    key = str(model_path).replace(str(REPO) + "/", "")
    return CLIFF_BUDGET.get(key, 1024)  # default conservative


def check_cliff_budget(prompt: str, model_path: str = None) -> tuple[bool, int, int]:
    """Return (within_budget, estimated_tokens, budget_tokens)."""
    model_path = model_path or str(DEFAULT_MODEL)
    est = estimate_tokens(prompt)
    budget = cliff_budget_for(model_path)
    return est <= budget, est, budget


class BudgetExceededError(Exception):
    """Raised when a stage tries to send a prompt larger than the cliff budget."""
    pass


# ----------------------------------------------------------------------------
# Server lifecycle
# ----------------------------------------------------------------------------
_server_proc: subprocess.Popen | None = None
_server_url: str | None = None
_server_model: str | None = None
_atexit_registered = False


def _port_in_use(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        s.connect((host, port))
        s.close()
        return True
    except (socket.error, ConnectionRefusedError):
        return False


def start_server(
    model: str | Path = DEFAULT_MODEL,
    *,
    binary: str | Path = DEFAULT_SERVER_BINARY,
    host: str = DEFAULT_SERVER_HOST,
    port: int = DEFAULT_SERVER_PORT,
    threads: int = 8,
    kv_type: str = "turbo_kv_4b",
    v_quant: str = "q4",
    startup_timeout: float = 180.0,
    verbose: bool = True,
) -> str:
    """Start a long-running quant-server. Returns the base URL."""
    global _server_proc, _server_url, _server_model, _atexit_registered

    if _server_proc is not None and _server_proc.poll() is None:
        if verbose:
            print(f"[server] already running at {_server_url}")
        return _server_url

    # Validate model and binary exist before starting
    if not Path(model).exists():
        raise FileNotFoundError(f"Model not found: {model}")
    if not Path(binary).exists():
        raise FileNotFoundError(f"Server binary not found: {binary}")

    # Register atexit handler to clean up server process on exit
    if not _atexit_registered:
        atexit.register(stop_server)
        _atexit_registered = True

    # Pick an unused port
    while _port_in_use(host, port):
        port += 1

    # Build command — unified server only supports -p and -j (no -k/-v/-H)
    is_unified = str(Path(binary).name).startswith("quant-server-unified")
    if is_unified:
        cmd = [str(binary), str(model), "-p", str(port), "-j", str(threads)]
    else:
        cmd = [
            str(binary), str(model),
            "-p", str(port), "-H", host,
            "-j", str(threads), "-k", kv_type, "-v", v_quant,
        ]
    if verbose:
        print(f"[server] starting: {' '.join(cmd)}")

    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"

    _server_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
    )
    _server_url = f"http://{host}:{port}"
    _server_model = str(model)

    # Wait for server to come up
    deadline = time.time() + startup_timeout
    last_err = None
    while time.time() < deadline:
        if _server_proc.poll() is not None:
            output = _server_proc.stdout.read() if _server_proc.stdout else ""
            raise RuntimeError(f"server died during startup:\n{output[-2000:]}")
        try:
            req = urllib.request.Request(f"{_server_url}/v1/models")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    if verbose:
                        elapsed = startup_timeout - (deadline - time.time())
                        print(f"[server] ready at {_server_url} after {elapsed:.1f}s")
                    return _server_url
        except (urllib.error.URLError, ConnectionRefusedError, socket.timeout) as e:
            last_err = e
            time.sleep(0.5)

    stop_server()
    raise RuntimeError(f"server did not start within {startup_timeout}s: {last_err}")


def stop_server():
    """Stop the long-running server (call from atexit or test teardown)."""
    global _server_proc, _server_url, _server_model
    if _server_proc is not None:
        try:
            _server_proc.terminate()
            _server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        _server_proc = None
        _server_url = None
        _server_model = None


# A short direct-style system prompt. Empirically this is the only thing
# that suppresses Llama-3.2-3B-Q4's tendency to emit "## Step 1: ..."
# reasoning chains in chat mode. Verified with the Acme test doc:
# without this, the model picks the first entity (primacy bias);
# with this, it correctly identifies the requested role.
DEFAULT_SYSTEM_PROMPT = "Answer in one short sentence. No reasoning steps."


MAX_LLM_RETRIES = 2  # retry once on transient server errors


def _check_server_alive() -> bool:
    """Check if the server process is still running (J11: crash detection)."""
    if _server_proc is None:
        return False
    return _server_proc.poll() is None


def _restart_server_if_dead(model: str | Path = DEFAULT_MODEL, verbose: bool = True):
    """Auto-restart server if it crashed (J4/J11: recovery)."""
    global _server_url
    if _server_proc is not None and _server_proc.poll() is not None:
        exit_code = _server_proc.returncode
        if verbose:
            print(f"[server] crashed (exit code {exit_code}), restarting...")
        stop_server()  # clean up
        start_server(model=model, verbose=verbose)


def llm_call(
    prompt: str,
    *,
    max_tokens: int = 16,
    temperature: float = 0.0,
    model: str | Path = DEFAULT_MODEL,
    enforce_budget: bool = True,
    system: str = DEFAULT_SYSTEM_PROMPT,
) -> LLMResult:
    """Run one quant.cpp inference call via the long-running server.

    The cliff invariant is enforced when enforce_budget=True (default):
    if the estimated prompt size exceeds the model's measured cliff
    budget, raises BudgetExceededError BEFORE invoking the model.

    Resilience features (audit batch 1):
    - Auto-restart server if it crashed between calls (J11)
    - Retry once on transient network errors (B2)
    - Distinguish network vs server vs timeout errors (B2)
    """
    global _server_url

    if enforce_budget:
        within, est, budget = check_cliff_budget(prompt, str(model))
        if not within:
            raise BudgetExceededError(
                f"Prompt estimated at {est} tokens exceeds cliff budget {budget} "
                f"for {model}. Either chunk the prompt or use a model with a "
                f"larger working memory."
            )

    # Validate max_tokens
    if max_tokens <= 0:
        max_tokens = 16

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")

    last_error = None
    for attempt in range(MAX_LLM_RETRIES + 1):
        # Lazy start or auto-restart if crashed (J4, J11)
        if _server_url is None:
            start_server(model=model)
        _restart_server_if_dead(model=model)

        req = urllib.request.Request(
            f"{_server_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break  # success
        except urllib.error.HTTPError as e:
            elapsed = time.time() - t0
            # 429 = server busy (retryable), others = server error
            if e.code == 429 and attempt < MAX_LLM_RETRIES:
                last_error = e
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s
                continue
            return LLMResult(text=f"[ERROR: HTTP {e.code}: {e.reason}]",
                             raw=str(e), n_tokens=0, elapsed=elapsed, is_error=True)
        except (ConnectionResetError, ConnectionRefusedError) as e:
            # Server likely crashed — try restart (B13)
            elapsed = time.time() - t0
            if attempt < MAX_LLM_RETRIES:
                last_error = e
                _restart_server_if_dead(model=model)
                continue
            return LLMResult(text=f"[ERROR: server connection lost: {e}]",
                             raw=str(e), n_tokens=0, elapsed=elapsed, is_error=True)
        except TimeoutError as e:
            elapsed = time.time() - t0
            return LLMResult(text=f"[ERROR: timeout after {elapsed:.0f}s]",
                             raw=str(e), n_tokens=0, elapsed=elapsed, is_error=True)
        except (urllib.error.URLError, OSError) as e:
            elapsed = time.time() - t0
            if attempt < MAX_LLM_RETRIES:
                last_error = e
                time.sleep(1)
                continue
            return LLMResult(text=f"[ERROR: network: {e}]",
                             raw=str(e), n_tokens=0, elapsed=elapsed, is_error=True)
    else:
        # All retries exhausted
        elapsed = time.time() - t0
        return LLMResult(text=f"[ERROR: {MAX_LLM_RETRIES+1} attempts failed: {last_error}]",
                         raw=str(last_error), n_tokens=0, elapsed=elapsed, is_error=True)
    elapsed = time.time() - t0

    # Robust JSON response parsing — handle malformed/incomplete responses
    text = ""
    n_tokens = 0
    is_error = False
    try:
        choices = payload.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            msg = choices[0].get("message") or choices[0].get("delta") or {}
            text = (msg.get("content") or "").strip()
        usage = payload.get("usage")
        if usage and isinstance(usage, dict):
            n_tokens = usage.get("completion_tokens", 0)
    except (KeyError, TypeError, IndexError, AttributeError):
        is_error = True
        text = f"[ERROR: malformed response: {str(payload)[:200]}]"

    if not text and not is_error:
        # Server returned empty content — likely state corruption.
        # Restart server to get a clean state for next call.
        is_error = True
        text = "[ERROR: empty response from server]"
        if _server_proc is not None:
            stop_server()
            # Next call will auto-restart via lazy start

    return LLMResult(text=text, raw=json.dumps(payload) if isinstance(payload, dict) else str(payload),
                     n_tokens=n_tokens, elapsed=elapsed, is_error=is_error)
