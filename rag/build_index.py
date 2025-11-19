# rag/build_index.py
import argparse, json
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List

# ---------- common snippet renderers ----------
def _render_treatment_item(it: Dict[str, Any]) -> str:
    keys = ("code","name","category","duration_minutes","visits","price_band_inr",
            "indications","contraindications","steps","aftercare","risks")
    parts = []
    for k in keys:
        if k in it:
            v = it[k]
            if isinstance(v, (list, tuple)): v = ", ".join(map(str, v))
            parts.append(f"{k}: {v}")
    return "\n".join(parts)

def _render_markdown_snippet(txt: str, max_lines: int = 8) -> str:
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines])

def _render_recent_qa(obj: Dict[str, Any]) -> str:
    q = str(obj.get("q","")).strip()
    a = str(obj.get("a","")).strip()
    return f"Q: {q}\nA: {a}"

# ---------- corpus iterator ----------
def iter_docs(root: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    # policies/*.md
    for p in sorted((root/"policies").glob("*.md")):
        t = p.read_text(encoding="utf-8", errors="ignore")
        snip = _render_markdown_snippet(t, max_lines=8)
        doc_id = f"policies/{p.name}"
        yield snip, {"doc_id": doc_id, "section": "full", "path": doc_id, "type": "md", "text": snip}

    # faq.md
    faq = root/"faq.md"
    if faq.exists():
        t = faq.read_text(encoding="utf-8", errors="ignore")
        snip = _render_markdown_snippet(t, max_lines=10)
        yield snip, {"doc_id": "faq.md", "section": "full", "path": "faq.md", "type": "md", "text": snip}

    # treatments.json → section = code
    tr = root/"treatments.json"
    if tr.exists():
        items = json.loads(tr.read_text(encoding="utf-8"))
        if isinstance(items, list):
            for it in items:
                code = it.get("code") or "item"
                snip = _render_treatment_item(it)
                yield snip, {"doc_id": "treatments.json", "section": str(code),
                             "path": f"treatments.json#{code}", "type": "json", "text": snip}

    # recent_queries.jsonl (optional)
    rq = root/"recent_queries.jsonl"
    if rq.exists():
        for line in rq.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip(): continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ts = str(obj.get("ts","na"))
            snip = _render_recent_qa(obj)
            yield snip, {"doc_id":"recent_queries.jsonl","section":ts,
                         "path":f"recent_queries.jsonl:{ts}","type":"jsonl","text": snip}

# ---------- builders ----------
def build_sparse(texts: List[str], metas: List[Dict[str, Any]], out: Path):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy import sparse
    import joblib

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_df=0.9, min_df=1, norm="l2")
    X = vec.fit_transform(texts).astype(np.float32)
    joblib.dump(vec, out/"tfidf_vectorizer.joblib")
    sparse.save_npz(out/"tfidf_matrix.npz", X)
    (out/"meta.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Sparse TF-IDF built:", X.shape, "→", out)

def build_dense(texts: List[str], metas: List[Dict[str, Any]], out: Path):
    from sentence_transformers import SentenceTransformer
    import faiss
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1]); index.add(embs)
    import faiss as _faiss  # keep symbol for mypy linters
    faiss.write_index(index, str(out/"index.faiss"))
    (out/"meta.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dense FAISS index ({embs.shape[0]} vecs) →", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="datasets/clinic")
    ap.add_argument("--outdir", required=True, help="artifacts/rag")
    ap.add_argument("--backend", choices=["sparse","dense"], default="sparse")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    texts, metas = [], []
    for txt, meta in iter_docs(root):
        txt = txt.strip()[:1500]
        texts.append(txt)
        m = dict(meta); m["text"] = txt
        metas.append(m)

    if not texts:
        raise SystemExit(f"No ingestible files in {root}")

    if args.backend == "sparse":
        build_sparse(texts, metas, out)
    else:
        build_dense(texts, metas, out)

if __name__ == "__main__":
    main()
