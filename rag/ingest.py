"""PDF -> cleaned page text -> overlapping chunks (with document and page number).

Run:  python -m rag.ingest        (writes data/index/chunks.jsonl)
"""
import json
import re
from collections import Counter

import pymupdf

from . import config

# Friendly names shown in citations. Anything not listed falls back to the file name.
DOC_TITLES = {
    "BAPLIE3.1.1-02.pdf": "BAPLIE 3.1.1 (bayplan / stowage plan)",
    "Why-BAPLIE-3-2024-0.4.pdf": "Why change to BAPLIE 3 (2024)",
    "SMDG_MIG_MOVINS_D95B_2.1.2.pdf": "MOVINS 2.1.2 (stowage instruction)",
    "SMDG-COPRAR213.pdf": "COPRAR 2.1.3 (container discharge/loading order)",
    "SMDG-COARRI213.pdf": "COARRI 2.1.3 (container discharge/loading report)",
    "SMDG-COPARN213.pdf": "COPARN 2.1.3 (container announcement)",
    "SMDG-CODECO213.pdf": "CODECO 2.1.3 (container gate-in/gate-out report)",
    "COREOR20.pdf": "COREOR 2.0 (container release order)",
    "COPINO20.pdf": "COPINO 2.0 (container pre-notification)",
    "SMDG_MIG_VERMAS_D24A_1.1.0.pdf": "VERMAS 1.1.0 (verified gross mass)",
    "IFTSTA-SMDG.pdf": "IFTSTA 1.0 (transport status)",
    "TPFREP41_20210525.pdf": "TPFREP 4.1 (terminal performance report)",
    "DESTIM-D96B-Equipment-Damage-and-Repair-Estimate-Message-Implementation-Guide-SMDG.pdf": "DESTIM (equipment damage and repair estimate)",
    "SMDG-Recommendation-01.pdf": "SMDG Recommendation 01 (tier numbering, 10+ tiers on deck)",
    "SMDG-Recommendation-02.pdf": "SMDG Recommendation 02 (containers with non-ISO identification)",
    "SMDG-Recommendation-03.pdf": "SMDG Recommendation 03 (COREOR and COPARN for container release)",
    "Recommendation-06-SMDG-Recommended-Messages-v1-20200220.pdf": "SMDG Recommendation 06 (recommended standard messages)",
    "Recommendation-07-UsageOfCodeLists-v8-20231112.pdf": "SMDG Recommendation 07 (usage of code lists)",
    "SMDG-Recommendation-08-MT-Tank-with-Residue_v8.pdf": "SMDG Recommendation 08 (empty tanks with residue)",
}


def _norm_line(line):
    """Line key used to spot running headers/footers: digits removed (page numbers, dates)."""
    return re.sub(r"\d+", "#", line.strip().lower())


def read_pages(path):
    doc = pymupdf.open(path)
    pages = [p.get_text() for p in doc]
    # A line that appears on more than half of the pages is a running header or footer.
    # Only lines with some real text qualify: short table cells ("M", "C", "1", "9999")
    # also repeat on most pages, but they are content, not headers.
    counts = Counter()
    for t in pages:
        counts.update({_norm_line(l) for l in t.splitlines() if l.strip()})
    limit = max(3, len(pages) * 0.5)
    boiler = {
        k for k, c in counts.items()
        if c > limit and len(k) >= 10 and sum(ch.isalpha() for ch in k) >= 4
    }

    cleaned = []
    for i, t in enumerate(pages, start=1):
        lines = [l.strip() for l in t.splitlines()]
        lines = [l for l in lines if l and _norm_line(l) not in boiler]
        text = "\n".join(lines)
        # PDF tables come out one cell per line; join short cells into rows so that
        # "EQD", "Equipment Details", "M", "1" stay together in one chunk.
        text = re.sub(r"\n(?=\S{1,25}\n)", " ", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        cleaned.append((i, text))
    return cleaned


def chunk_text(text, size=config.CHUNK_CHARS, overlap=config.CHUNK_OVERLAP):
    """Split on paragraph/line/sentence boundaries into chunks of about `size` characters."""
    if len(text) <= size:
        return [text] if text else []
    pieces = re.split(r"(?<=\n)|(?<=\. )", text)
    chunks, cur = [], ""
    for p in pieces:
        if len(cur) + len(p) > size and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:]  # carry the tail over so a fact split at the edge survives
        while len(p) > size:  # one huge line (long table row): hard split
            chunks.append((cur + p[: size - len(cur)]).strip())
            p = p[size - len(cur):]
            cur = ""
        cur += p
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 40]


def build_chunks(raw_dir=config.RAW_DIR):
    chunks = []
    for path in sorted(raw_dir.glob("*.pdf")):
        title = DOC_TITLES.get(path.name, path.stem)
        for page, text in read_pages(path):
            for j, c in enumerate(chunk_text(text)):
                chunks.append(
                    {
                        "id": f"{path.stem}#p{page}-{j}",
                        "doc": path.name,
                        "title": title,
                        "page": page,
                        "text": c,
                    }
                )
    return chunks


def main():
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks()
    with open(config.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    docs = len({c["doc"] for c in chunks})
    print(f"{len(chunks)} chunks from {docs} documents -> {config.INDEX_DIR / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
