"""Offline checks (no Ollama/Gemini needed). Run: python test_toolkit.py  (or pytest)."""
import json
import tempfile
from pathlib import Path

import pymupdf as fitz
import numpy as np

import mcp_rag
import orchestrator as orc


def test_chunk():
    assert mcp_rag.chunk("abc", 10, 2) == ["abc"]
    parts = mcp_rag.chunk("x" * 25, 10, 3)
    assert all(len(p) <= 10 for p in parts) and parts[1] == "x" * 10 and len(parts) == 4
    assert mcp_rag.chunk("   ", 10, 2) == []


def test_route():
    assert orc.route("Summarize the thesis") == "summarizer"
    assert orc.route("extract all dates as JSON") == "extractor"
    assert orc.route("research and compare the two methods") == "research"
    assert orc.route("what is the learning rate?") == "qa"


def test_parse_action():
    assert orc.parse_action('Sure! {"tool": "search_pdfs", "args": {"query": "x"}}')["tool"] == "search_pdfs"
    assert orc.parse_action('{"answer": "42"}') == {"answer": "42"}
    assert orc.parse_action("plain text reply") == {"answer": "plain text reply"}


def test_fallback():
    def down(p, t):
        raise orc.ModelError("down")
    orc.MODELS.update(gemini=down, ollama=lambda p, t: "from ollama")
    assert orc.complete("hi") == "from ollama"
    orc.MODELS["ollama"] = down
    try:
        orc.complete("hi")
        assert False, "should raise"
    except orc.ModelError as e:
        assert "gemini#1" in str(e) and "ollama#1" in str(e)

    def missing(p, t):
        raise orc.ModelUnavailable("not installed")
    calls = []
    orc.MODELS.update(gemini=missing, ollama=lambda p, t: calls.append(1) or "ok")
    assert orc.complete("hi") == "ok" and len(calls) == 1  # missing binary: no retries, straight to fallback


def test_index_search_roundtrip():
    def fake_embed(texts, prefix):  # deterministic bag-of-letters vectors
        v = np.array([[t.lower().count(c) for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts], "float32") + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    mcp_rag.embed = fake_embed
    with tempfile.TemporaryDirectory() as d:
        mcp_rag.KB_DIR = Path(d) / "kb"
        pdf = Path(d) / "doc.pdf"
        doc = fitz.open()
        for text in ["zebra zoo zigzag", "apple banana cabbage"]:
            doc.new_page().insert_text((72, 72), text)
        doc.save(pdf)
        doc.close()
        (Path(d) / "corrupt.pdf").write_bytes(b"not a pdf")
        out = mcp_rag.index_pdfs(d)
        assert "2 chunks" in out and "corrupt.pdf" in out  # bad file skipped, good one indexed
        assert "2 chunks" in mcp_rag.index_pdfs(str(pdf))  # re-index replaces, doesn't duplicate
        hit = json.loads(mcp_rag.search_pdfs("zzz zoo", k=1))[0]
        assert hit["page"] == 1 and hit["file"] == "doc.pdf"
        assert json.loads(mcp_rag.list_pdfs())[str(pdf.resolve())] == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
