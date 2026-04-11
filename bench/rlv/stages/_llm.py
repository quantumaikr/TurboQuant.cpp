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
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MODEL = REPO / "models" / "Llama-3.2-3B-Instruct-Q8_0.gguf"
DEFAULT_SERVER_BINARY = REPO / "build_metal" / "quant-server"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8421  # arbitrary, avoid conflicts with 8080

# Phase 1B cliff measurements. NEVER set a stage prompt larger than this.
CLIFF_BUDGET = {
    "models/Llama-3.2-3B-Instruct-Q8_0.gguf": 1024,
    "models/Llama-3.2-1B-Instruct-Q8_0.gguf": 512,
}


@dataclass
class LLMResult:
    text: str          # the model's generated text (between --- delimiters)
    raw: str           # the full CLI stdout+stderr
    n_tokens: int      # generated token count
    elapsed: float     # wall seconds


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
    startup_timeout: float = 120.0,
    verbose: bool = True,
) -> str:
    """Start a long-running quant-server. Returns the base URL."""
    global _server_proc, _server_url, _server_model

    if _server_proc is not None and _server_proc.poll() is None:
        if verbose:
            print(f"[server] already running at {_server_url}")
        return _server_url

    # Pick an unused port
    while _port_in_use(host, port):
        port += 1

    cmd = [
        str(binary), str(model),
        "-p", str(port),
        "-H", host,
        "-j", str(threads),
        "-k", kv_type,
        "-v", v_quant,
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


def llm_call(
    prompt: str,
    *,
    max_tokens: int = 64,
    temperature: float = 0.0,
    model: str | Path = DEFAULT_MODEL,
    enforce_budget: bool = True,
    system: str = DEFAULT_SYSTEM_PROMPT,
) -> LLMResult:
    """Run one quant.cpp inference call via the long-running server.

    The cliff invariant is enforced when enforce_budget=True (default):
    if the estimated prompt size exceeds the model's measured cliff
    budget, raises BudgetExceededError BEFORE invoking the model.
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

    # Lazy server start if no server is running yet
    if _server_url is None:
        start_server(model=model)

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
    req = urllib.request.Request(
        f"{_server_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        elapsed = time.time() - t0
        return LLMResult(text=f"[ERROR: {e}]", raw=str(e), n_tokens=0, elapsed=elapsed)
    elapsed = time.time() - t0

    text = ""
    n_tokens = 0
    if "choices" in payload and payload["choices"]:
        msg = payload["choices"][0].get("message", {})
        text = msg.get("content", "").strip()
    if "usage" in payload:
        n_tokens = payload["usage"].get("completion_tokens", 0)

    return LLMResult(text=text, raw=json.dumps(payload), n_tokens=n_tokens, elapsed=elapsed)
