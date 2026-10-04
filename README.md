# Local AI Agent Toolkit

Implements [`PLAN.md`](PLAN.md) (all-Python variant).

- `mcp_rag.py`: FastMCP server. PDFs → PyMuPDF → chunks → `nomic-embed-text` (Ollama) → FAISS in `kb/`. Tools: `index_pdfs`, `search_pdfs`, `list_pdfs`.
- `orchestrator.py`: keyword router → agent (research / summarizer / extractor / qa) → JSON tool-use loop over MCP. Uses Gemini CLI first and falls back to Ollama, with retries and a logged reason for each failure.

```powershell
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
ollama pull nomic-embed-text; ollama pull qwen2.5-coder:3b
copy .env.example .env      # optional; MODEL_ORDER=ollama skips Gemini

.\.venv\Scripts\python orchestrator.py "index the pdfs in D:\papers"
.\.venv\Scripts\python orchestrator.py "what loss function does the paper use?" -v
.\.venv\Scripts\python test_toolkit.py     # offline tests, no models needed
```

Use with other MCP clients (e.g. Claude Code): `claude mcp add kb -- <path>\.venv\Scripts\python.exe <path>\mcp_rag.py`
