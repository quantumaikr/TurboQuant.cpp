#!/usr/bin/env python3
"""Phase A-2: Large document stress test — 1.3MB wikitext (860+ chunks).

This tests whether RLV scales to REAL document sizes. The original
eval_wikitext.py uses ppl_8k.txt (35K chars, 23 chunks, 3 articles).
This uses wikitext2_test.txt (1.3MB, ~860 chunks, 63 articles).

Key challenges at this scale:
- 860 chunks → locator must discriminate among 37x more candidates
- 63 articles with overlapping topics (military history, sports, poetry)
- BM25 IDF changes: common words have lower discrimination power
- Multiple articles about similar subjects (battles, sports figures)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlv_orchestrator import answer_question
from stages import _llm
from stages import gist as gist_stage

DOC_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "wikitext2_test.txt"

# 20 questions across 10+ different articles — diverse topics, difficulty levels
QUESTIONS = [
    # === Already-tested articles (Boulter, Du Fu, Kiss You) — regression check ===
    {"id": 1, "topic": "boulter", "type": "single-hop",
     "question": "Who directed the production of Mercury Fur in which Boulter appeared?",
     "fragments": ["john tiffany", "tiffany"]},
    {"id": 2, "topic": "dufu", "type": "single-hop",
     "question": "In what year did Du Fu first meet Li Bai?",
     "fragments": ["744"]},
    {"id": 3, "topic": "kiss_you", "type": "single-hop",
     "question": "Who directed the Kiss You music video?",
     "fragments": ["vaughan arnell", "arnell"]},

    # === Military history (Ise-class, Naktong, Ironclad) ===
    {"id": 4, "topic": "ise_class", "type": "single-hop",
     "question": "What disaster did the Ise-class battleships carry supplies for in 1923?",
     "fragments": ["earthquake", "kanto", "kantō"]},
    {"id": 5, "topic": "ise_class", "type": "multi-hop",
     "question": "After which battle were the Ise-class ships rebuilt with a flight deck?",
     "fragments": ["midway"]},
    {"id": 6, "topic": "ironclad", "type": "single-hop",
     "question": "What was the name of the first ironclad battleship launched in 1859?",
     "fragments": ["gloire"]},
    {"id": 7, "topic": "naktong", "type": "single-hop",
     "question": "The Second Battle of Naktong Bulge was part of which larger battle?",
     "fragments": ["pusan perimeter", "pusan"]},

    # === Sports (Dick Rifenburg, Clayton Kershaw, Ben Amos) ===
    {"id": 8, "topic": "rifenburg", "type": "single-hop",
     "question": "What NFL team did Dick Rifenburg play for in 1950?",
     "fragments": ["detroit lions", "detroit", "lions"]},
    {"id": 9, "topic": "kershaw", "type": "single-hop",
     "question": "In what year was Clayton Kershaw drafted?",
     "fragments": ["2006"]},
    {"id": 10, "topic": "kershaw", "type": "single-hop",
     "question": "On what date did Clayton Kershaw pitch a no-hitter?",
     "fragments": ["june 18", "2014"]},
    {"id": 11, "topic": "amos", "type": "single-hop",
     "question": "Which Manchester United academy did Ben Amos join from?",
     "fragments": ["crewe", "crewe alexandra"]},

    # === Science/Weather (Dvorak technique, Hurricane, Temnospondyli) ===
    {"id": 12, "topic": "dvorak", "type": "single-hop",
     "question": "Who developed the Dvorak technique for estimating tropical cyclone intensity?",
     "fragments": ["vernon dvorak", "vernon", "dvorak"]},
    {"id": 13, "topic": "hurricane", "type": "single-hop",
     "question": "Where did the 1933 Treasure Coast hurricane make landfall in Florida?",
     "fragments": ["jupiter"]},
    {"id": 14, "topic": "hurricane", "type": "single-hop",
     "question": "What were the peak winds of the 1933 Treasure Coast hurricane?",
     "fragments": ["140"]},
    {"id": 15, "topic": "temno", "type": "single-hop",
     "question": "What does the Greek word 'temnein' mean in the name Temnospondyli?",
     "fragments": ["cut"]},

    # === Literature/Arts (Imagism, Little Gidding, Portage) ===
    {"id": 16, "topic": "imagism", "type": "single-hop",
     "question": "Imagism is considered the first organized movement of what literary period?",
     "fragments": ["modernist", "modernism"]},
    {"id": 17, "topic": "gidding", "type": "single-hop",
     "question": "Little Gidding is the fourth poem in which series by T.S. Eliot?",
     "fragments": ["four quartets", "quartets"]},
    {"id": 18, "topic": "portage", "type": "multi-hop",
     "question": "Who wrote The Portage to San Cristobal of A.H., a novella about Nazi hunters?",
     "fragments": ["george steiner", "steiner"]},

    # === Geography/Infrastructure (NY Route 31B, Osbert de Bayeux) ===
    {"id": 19, "topic": "route", "type": "single-hop",
     "question": "NY State Route 31B connected Weedsport to which route?",
     "fragments": ["ny 5", "5"]},
    {"id": 20, "topic": "osbert", "type": "single-hop",
     "question": "In which diocese was Osbert de Bayeux an archdeacon?",
     "fragments": ["york"]},
]


def fuzzy_hit(text, fragments):
    t = text.lower()
    matched = [f for f in fragments if f in t]
    return (len(matched) > 0, matched)


def collect_text(result):
    parts = [result.get("final_answer", "")]
    for a in result.get("research", {}).get("attempts", []):
        parts.append(a.get("answer", "") or "")
    return " ".join(parts).lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--only", type=int, default=None)
    args = parser.parse_args()

    doc_text = DOC_PATH.read_text(encoding="utf-8", errors="replace")
    print("=" * 76)
    print("Phase A-2: LARGE DOCUMENT stress test (1.3MB wikitext, 860+ chunks)")
    print("=" * 76)
    print(f"Document: {DOC_PATH.name}")
    print(f"  chars: {len(doc_text):,}")
    print(f"  est tokens: ~{len(doc_text)//3:,}")
    print(f"  articles: 63")
    print(f"  questions: {len(QUESTIONS)} across 13 topics")
    print("-" * 76)

    _llm.start_server()
    t_start = time.time()
    try:
        print("[setup] building gist (one-time, no LLM)...")
        cached_gist = gist_stage.build_gist(doc_text, doc_id="wikitext2_full", verbose=False)
        print(f"[setup] gist has {len(cached_gist.chunks)} chunks")
        print()

        passed = 0
        results = []
        for q in QUESTIONS:
            if args.only is not None and q["id"] != args.only:
                continue
            print(f"--- Q{q['id']} ({q['topic']}, {q['type']}) ---")
            print(f"Q: {q['question']}")
            t_q = time.time()
            try:
                r = answer_question(doc_text, q["question"], doc_id="wt2_full",
                                    cached_gist=cached_gist, verbose=args.verbose)
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"q": q, "ok": False})
                continue
            elapsed = time.time() - t_q
            scoring_text = collect_text(r)
            ok, matched = fuzzy_hit(scoring_text, q["fragments"])
            mark = "PASS" if ok else "FAIL"
            if ok: passed += 1
            print(f"  [{mark}] ({elapsed:.1f}s) {r['final_answer'][:120]!r}")
            if ok:
                print(f"        matched: {matched}")
            else:
                print(f"        expected: {q['fragments']}")
            print()
            results.append({"q": q, "ok": ok, "elapsed": elapsed})

    finally:
        _llm.stop_server()

    total = time.time() - t_start
    n = len(results)
    print("=" * 76)
    print(f"RESULTS: {passed}/{n} in {total:.0f}s")
    print("=" * 76)
    for r in results:
        q = r["q"]
        mark = "OK" if r["ok"] else "XX"
        print(f"  {q['id']:>2} {q['topic']:<12} {q['type']:<10} {mark}")
    print()
    if passed == n:
        print(f"  LARGE DOC GATE: PASS ✅ ({passed}/{n})")
    else:
        print(f"  LARGE DOC GATE: {passed}/{n} ({100*passed/n:.0f}%)")
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
