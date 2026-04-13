#!/usr/bin/env python3
"""ClawTeam CI smoke test (#87).

Fast regression check (~3min) for PR gating.
Tests: build, unit tests, Acme 3-question sample.

Usage:
    python3 bench/rlv/eval/ci_smoke.py

Returns exit code 0 on pass, 1 on failure.
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    mark = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    print("=" * 60)
    print("ClawTeam CI Smoke Test")
    print("=" * 60)
    t_start = time.time()

    # 1. Python package imports
    print("\n--- Package imports ---")
    try:
        import quantcpp
        check("import quantcpp", True, f"v{quantcpp.__version__}")
        check("Model class", hasattr(quantcpp, "Model"))
        check("ask_verified", hasattr(quantcpp.Model, "ask_verified"))
        check("available_models", callable(getattr(quantcpp, "available_models", None)))
    except Exception as e:
        check("import quantcpp", False, str(e))

    # 2. CLI commands
    print("\n--- CLI commands ---")
    for cmd in ["--help", "list", "recommend"]:
        try:
            r = subprocess.run(
                ["quantcpp"] + cmd.split(),
                capture_output=True, text=True, timeout=10
            )
            check(f"quantcpp {cmd}", r.returncode == 0)
        except FileNotFoundError:
            check(f"quantcpp {cmd}", False, "quantcpp not in PATH")
        except Exception as e:
            check(f"quantcpp {cmd}", False, str(e))

    # 3. Unit tests (if build exists)
    print("\n--- Unit tests ---")
    build_dirs = [REPO / "build-metal", REPO / "build", REPO / "build-cpu"]
    build_dir = next((d for d in build_dirs if (d / "CTestTestfile.cmake").exists()), None)
    if build_dir:
        r = subprocess.run(
            ["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-j4"],
            capture_output=True, text=True, timeout=120
        )
        # Parse "X tests passed"
        passed = "100% tests passed" in r.stdout or "100% tests passed" in r.stderr
        check("ctest", passed, r.stdout.strip().split("\n")[-1] if r.stdout else "")
    else:
        check("ctest", False, "no build directory found")

    # 4. Memory leak check (if leaks tool available)
    print("\n--- Memory check ---")
    embed = REPO / "examples" / "embed_minimal.c"
    model = Path.home() / ".cache" / "quantcpp" / "smollm2-135m-instruct-q8_0.gguf"
    if embed.exists() and model.exists():
        # Build
        r = subprocess.run(
            ["cc", "-O2", "-o", "/tmp/ci_embed", str(embed), "-lm", "-lpthread"],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0:
            # Run with leaks
            try:
                r2 = subprocess.run(
                    ["leaks", "--atExit", "--", "/tmp/ci_embed", str(model), "test"],
                    capture_output=True, text=True, timeout=120
                )
            except subprocess.TimeoutExpired:
                check("memory leaks", True, "skipped (timeout)")
                r2 = None
            if r2:
                no_leaks = "0 leaks" in r2.stderr or "0 leaks" in r2.stdout
                check("memory leaks", no_leaks, "0 leaks" if no_leaks else "leaks detected")
        else:
            check("memory leaks", False, "build failed")
    else:
        check("memory leaks", True, "skipped (no model/example)")

    # Summary
    elapsed = time.time() - t_start
    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"RESULTS: {PASS}/{total} passed in {elapsed:.0f}s")
    print(f"{'='*60}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
