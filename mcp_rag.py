"""MCP-RAG server: PDFs -> chunks -> nomic-embed-text (Ollama) -> FAISS, exposed as MCP tools."""
import json
import os
from pathlib import Path

import faiss
import pymupdf as fitz
import httpx
import numpy as np
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
KB_DIR = Path(os.getenv("KB_DIR", HERE / "kb"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

mcp = FastMCP("mcp-rag", host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", 8765)))


def chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # ponytail: fixed-window char splitter; swap in RecursiveCharacterTextSplitter if sentence boundaries matter
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step) if text[i:i + size].strip()]


def embed(texts, prefix):
    """nomic-embed-text wants task prefixes: 'search_document: ' / 'search_query: '."""
    out = []
    for i in range(0, len(texts), 64):
        r = httpx.post(f"{OLLAMA_URL}/api/embed", timeout=300,
                       json={"model": EMBED_MODEL, "input": [prefix + t for t in texts[i:i + 64]]})
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    v = np.array(out, dtype="float32")
    faiss.normalize_L2(v)  # inner product on unit vectors = cosine similarity
    return v


def load_kb():
    if (KB_DIR / "index.faiss").exists():
        return faiss.read_index(str(KB_DIR / "index.faiss")), json.loads((KB_DIR / "metadata.json").read_text("utf-8"))
    return None, []


def save_kb(index, meta):
    KB_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(KB_DIR / "index.faiss"))
    (KB_DIR / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False), "utf-8")


@mcp.tool()
def index_pdfs(path: str) -> str:
    """Index a PDF file, or every PDF under a directory. Re-indexing a file replaces its old chunks."""
    p = Path(path).expanduser()
    files = [p] if p.is_file() else sorted(p.rglob("*.pdf")) if p.is_dir() else []
    if not files:
        return f"No PDFs found at {path}"
    index, meta = load_kb()
    sources = {str(f.resolve()) for f in files}
    keep = [i for i, m in enumerate(meta) if m["source"] not in sources]

    new_meta, skipped = [], []
    for f in files:
        try:
            with fitz.open(f) as doc:
                for page_no, page in enumerate(doc, 1):
                    new_meta += [{"source": str(f.resolve()), "page": page_no, "text": c} for c in chunk(page.get_text())]
        except fitz.FileDataError:  # one bad file shouldn't sink the whole folder
            skipped.append(f.name)
    note = f" Skipped unreadable: {skipped}." if skipped else ""
    if not new_meta:
        return "No extractable text in those PDFs (scanned images need OCR first)." + note

    vecs = embed([m["text"] for m in new_meta], "search_document: ")
    if keep:  # ponytail: full rebuild on every ingest, fine for a local KB up to ~100k chunks
        vecs = np.vstack([np.stack([index.reconstruct(i) for i in keep]), vecs])
    new_index = faiss.IndexFlatIP(vecs.shape[1])
    new_index.add(vecs)
    save_kb(new_index, [meta[i] for i in keep] + new_meta)
    return (f"Indexed {len(files) - len(skipped)} PDF(s) into {len(new_meta)} chunks. "
            f"KB total: {new_index.ntotal} chunks.{note}")


@mcp.tool()
def search_pdfs(query: str, k: int = 5) -> str:
    """Semantic search over indexed PDFs. Returns JSON list of {file, page, score, text}."""
    index, meta = load_kb()
    if index is None or index.ntotal == 0:
        return "Knowledge base is empty. Call index_pdfs first."
    scores, ids = index.search(embed([query], "search_query: "), min(k, index.ntotal))
    return json.dumps([{"file": Path(meta[i]["source"]).name, "page": meta[i]["page"],
                        "score": round(float(s), 3), "text": meta[i]["text"]}
                       for s, i in zip(scores[0], ids[0])], ensure_ascii=False)


@mcp.tool()
def list_pdfs() -> str:
    """List indexed PDFs with their chunk counts."""
    counts = {}
    for m in load_kb()[1]:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return json.dumps(counts, indent=1) if counts else "Knowledge base is empty."


if __name__ == "__main__":
    mcp.run("streamable-http" if os.getenv("MCP_TRANSPORT") == "http" else "stdio")
