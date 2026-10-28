"""Evaluate retrieval and answers on eval/questions.jsonl.

  python -m eval.run_eval retrieval --no-llm     # no API key needed
  python -m eval.run_eval retrieval              # all rewrite policies x search modes
  python -m eval.run_eval answers --model claude-haiku-5-5 --rewrite auto

Retrieval is scored at page level: a hit means a retrieved chunk comes from a gold
(document, page). Answers are graded by a separate Claude call against the
reference facts, and citations are checked against the gold pages.
"""
import argparse
import json
import re
import time
from pathlib import Path

from rag import config
from rag.index import Index
from rag.retrieve import search

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def load_questions():
    with open(HERE / "questions.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def is_gold(chunk, gold):
    return any(chunk["doc"] == g["doc"] and chunk["page"] in g["pages"] for g in gold)


def first_hit_rank(chunks, gold):
    for r, c in enumerate(chunks, 1):
        if is_gold(c, gold):
            return r
    return None


# ----------------------------------------------------------------------------- retrieval
POLICIES = ("none", "always", "auto", "fuse")


def eval_retrieval(k=config.TOP_K, use_llm=True):
    """Hit@k and MRR@10 for every rewrite policy x search mode.

    Each question is rewritten once and the rewrite is reused, so policies are compared
    on exactly the same rewritten queries.
    """
    qs = [q for q in load_questions() if q["gold"]]
    index = Index()
    policies = POLICIES if use_llm else ("none",)
    rag = None
    if use_llm:
        from rag.answer import RAG
        rag = RAG(index=index)
        cache = {}
        orig = rag.rewrite

        def cached(question):
            if question not in cache:
                cache[question] = orig(question)
            return cache[question]
        rag.rewrite = cached

    rows = []
    for q in qs:
        row = {"id": q["id"], "lang": q["lang"]}
        for pol in policies:
            for mode in ("dense", "sparse", "hybrid"):
                if rag:
                    got, query, kw, _ = rag.retrieve(q["question"], k=10, mode=mode, rewrite=pol)
                    row["rewritten"] = rag.rewrite(q["question"])[:2]
                else:
                    got = search(index, q["question"], k=10, mode=mode)
                row[f"{pol}/{mode}"] = first_hit_rank(got, q["gold"])
        rows.append(row)

    def summary(sub, key):
        n = len(sub)
        return (sum(1 for r in sub if r[key] and r[key] <= k) / n,
                sum(1 / r[key] for r in sub if r[key]) / n)

    out = {"k": k, "n": len(rows), "table": {}, "rows": rows}
    print(f"\nRetrieval: hit@{k} / MRR@10 (page level), {len(rows)} answerable questions")
    print(f"{'policy/mode':16} {'all':>12} {'English':>12} {'Korean':>12}")
    en = [r for r in rows if r["lang"] == "en"]
    ko = [r for r in rows if r["lang"] == "ko"]
    for pol in policies:
        for mode in ("dense", "sparse", "hybrid"):
            key = f"{pol}/{mode}"
            res = {"all": summary(rows, key), "en": summary(en, key), "ko": summary(ko, key)}
            out["table"][key] = res
            print(f"{key:16} " + " ".join(f"{h:5.0%} / {m:.2f}" for h, m in res.values()))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


# ----------------------------------------------------------------------------- answers
JUDGE_PROMPT = """You grade an answer from a document Q&A assistant.

Question: {question}
Reference facts (from the source documents): {facts}
Answer to grade:
<<<
{answer}
>>>

Grade only against the reference facts (the answer may be in Korean; judge meaning).
- "correct": states all reference facts, nothing contradicting them
- "partial": states some reference facts, or is vague on one, with nothing contradicting them
- "wrong": misses the main fact, or states something that contradicts the reference facts

Return only JSON: {{"grade": "correct|partial|wrong", "reason": "<one sentence>"}}"""

REFUSAL_PROMPT = """A document Q&A assistant was asked a question whose answer is NOT in its \
documents. Did the assistant clearly say it could not find / the documents do not contain the \
answer (a refusal), or did it give an answer anyway (a hallucination)? Mentioning related \
information while clearly saying the specific answer is not in the documents counts as refusal.

Question: {question}
Answer:
<<<
{answer}
>>>

Return only JSON: {{"refused": true|false, "reason": "<one sentence>"}}"""


def judge(client, prompt, model):
    from rag.llm import create, text_of
    msg = create(client, model, max_tokens=300,
                 messages=[{"role": "user", "content": prompt}])
    m = re.search(r"\{.*\}", text_of(msg), re.S)
    return json.loads(m.group(0)), msg.usage


def eval_answers(model, judge_model, mode="hybrid", rewrite="auto", tag=None):
    from rag.answer import RAG
    rag = RAG()
    rows = []
    for q in load_questions():
        res = rag.ask(q["question"], mode=mode, rewrite=rewrite, model=model)
        row = {"id": q["id"], "lang": q["lang"], "answerable": bool(q["gold"]),
               "answer": res["answer"], "search_query": res["search_query"],
               "seconds": res["seconds"], "input_tokens": res["input_tokens"],
               "output_tokens": res["output_tokens"]}
        if q["gold"]:
            g, _ = judge(rag.client, JUDGE_PROMPT.format(
                question=q["question"], facts="; ".join(q["facts"]), answer=res["answer"]), judge_model)
            row["grade"], row["reason"] = g["grade"], g["reason"]
            cited = [s for s in res["sources"] if s["cited"]]
            row["n_cited"] = len(cited)
            row["cited_gold"] = any(is_gold(s, q["gold"]) for s in cited)
            row["retrieved_gold"] = any(is_gold(s, q["gold"]) for s in res["sources"])
        else:
            g, _ = judge(rag.client, REFUSAL_PROMPT.format(
                question=q["question"], answer=res["answer"]), judge_model)
            row["refused"], row["reason"] = bool(g["refused"]), g["reason"]
        rows.append(row)
        mark = row.get("grade") or ("refused" if row.get("refused") else "HALLUCINATED")
        print(f"{q['id']:5} {mark:12} {res['seconds']:5.1f}s  {q['question'][:60]}")
        time.sleep(0.5)

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    summ = {
        "model": model, "judge": judge_model, "mode": mode, "rewrite": rewrite,
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "correct": sum(r["grade"] == "correct" for r in ans),
        "partial": sum(r["grade"] == "partial" for r in ans),
        "wrong": sum(r["grade"] == "wrong" for r in ans),
        "correct_en": sum(r["grade"] == "correct" for r in ans if r["lang"] == "en"),
        "n_en": sum(r["lang"] == "en" for r in ans),
        "correct_ko": sum(r["grade"] == "correct" for r in ans if r["lang"] == "ko"),
        "n_ko": sum(r["lang"] == "ko" for r in ans),
        "cited_gold": sum(r["cited_gold"] for r in ans),
        "retrieved_gold": sum(r["retrieved_gold"] for r in ans),
        "refused": sum(r["refused"] for r in una),
        "median_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2],
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
    }
    print(json.dumps(summ, indent=1))
    RESULTS.mkdir(exist_ok=True)
    tag = tag or f"{model}_{mode}_{rewrite}"
    (RESULTS / f"answers_{tag}.json").write_text(
        json.dumps({"summary": summ, "rows": rows}, ensure_ascii=False, indent=1))
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "answers"])
    ap.add_argument("--rewrite", default="auto", choices=POLICIES)
    ap.add_argument("--no-llm", action="store_true", help="retrieval without query rewriting")
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--model", default=config.ANSWER_MODEL)
    ap.add_argument("--judge", default="claude-sonnet-5-5")
    ap.add_argument("--tag")
    a = ap.parse_args()
    if a.what == "retrieval":
        eval_retrieval(use_llm=not a.no_llm)
    else:
        eval_answers(a.model, a.judge, mode=a.mode, rewrite=a.rewrite, tag=a.tag)
