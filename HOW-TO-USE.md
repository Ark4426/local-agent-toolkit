# How to Use: Local AI Agent Toolkit & RAG Documents Q&A

Two local projects that answer questions from your own documents with local models (Ollama).

| Project | Folder | Use it when you want… |
|---|---|---|
| **Local AI Agent Toolkit** | `local-agent-toolkit\` | a command-line tool where *agents* decide how to search your PDFs (research, summarize, extract, Q&A) |
| **RAG Documents Q&A Service** | `rag-docs-qa\` | a REST API / web page where you upload PDF, DOCX, TXT or MD files and ask questions, with cited sources |

---

## 1. One-time setup

### 1.1 Prerequisites
- **Python 3.10+**: check with `python --version`
- **Ollama** running locally: https://ollama.com/download
- **Gemini CLI** (optional, toolkit only): if it's not installed, the toolkit falls back to Ollama

### 1.2 Pull the models
```powershell
ollama pull nomic-embed-text      # embeddings (both projects)
ollama pull qwen2.5-coder:3b      # answer model (default)
```
> Want better answers? Pull a bigger model (e.g. `ollama pull qwen2.5:7b`) and set it in `.env` (see §4).

### 1.3 Install each project
Both projects already have a `.venv` on this machine. On a fresh machine:
```powershell
cd D:\test\project\local-agent-toolkit
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

cd D:\test\project\rag-docs-qa
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

### 1.4 (Optional) configuration file
```powershell
copy .env.example .env     # run inside the project folder, then edit .env
```

---

## 2. Local AI Agent Toolkit

### 2.1 Basic usage
```powershell
cd D:\test\project\local-agent-toolkit
$env:MODEL_ORDER = "ollama"          # skip Gemini if it isn't installed (or put it in .env)

.\.venv\Scripts\python orchestrator.py "<your task>"
```
Add `-v` to see which agent was chosen and which tools it called.

### 2.2 Step 1: index your PDFs
Give it a single PDF or a whole folder (searched recursively):
```powershell
.\.venv\Scripts\python orchestrator.py "index the pdfs in D:\my\papers"
```
- Re-indexing a file **replaces** its old chunks (no duplicates).
- Corrupt PDFs are skipped and named in the result; scanned (image-only) PDFs have no text and need OCR first.
- The knowledge base is saved in `local-agent-toolkit\kb\`.

### 2.3 Step 2: ask things
```powershell
.\.venv\Scripts\python orchestrator.py "What is the peak efficiency of the inverter?"
.\.venv\Scripts\python orchestrator.py "summarize the solar handbook"
.\.venv\Scripts\python orchestrator.py "research and compare the safety and maintenance rules"
.\.venv\Scripts\python orchestrator.py "extract all leave entitlements as JSON"
.\.venv\Scripts\python orchestrator.py "list the indexed pdfs"
```

### 2.4 Which agent handles my task?
The task is routed by keywords. Override it with `--agent <name>`.

| Agent | Triggered by words like | Tools it may use | Max tool calls |
|---|---|---|---|
| `research` | research, investigate, compare, explore, analyze | search, list | 6 |
| `summarizer` | summarize, summary, overview, gist, tl;dr | search, list | 3 |
| `extractor` | extract, index, ingest, list, table, fields, json | index, search, list | 5 |
| `qa` | *(default for everything else)* | search | 3 |

```powershell
.\.venv\Scripts\python orchestrator.py "what does chapter 3 say about loss functions?" --agent research
```

### 2.5 Model order (Gemini → Ollama)
`MODEL_ORDER` sets which models are tried, in order. If one fails (not installed, timeout, error), the next is tried. If all fail, you get one line listing every reason.

| `MODEL_ORDER` | Behaviour |
|---|---|
| `gemini,ollama` *(default)* | Gemini CLI first, Ollama fallback |
| `ollama` | Ollama only |
| `gemini` | Gemini only |

### 2.6 Use the knowledge base from other MCP clients
`mcp_rag.py` is a standard MCP server with tools `index_pdfs`, `search_pdfs`, `list_pdfs`. For example, in Claude Code:
```powershell
claude mcp add kb -- D:\test\project\local-agent-toolkit\.venv\Scripts\python.exe D:\test\project\local-agent-toolkit\mcp_rag.py
```

**HTTP mode** (server and orchestrator as separate processes):
```powershell
# terminal 1
$env:MCP_TRANSPORT="http"; .\.venv\Scripts\python mcp_rag.py
# terminal 2
$env:MCP_TRANSPORT="http"; .\.venv\Scripts\python orchestrator.py "list the indexed pdfs"
```

---

## 3. RAG Documents Q&A Service

### 3.1 Start the server
```powershell
cd D:\test\project\rag-docs-qa
.\.venv\Scripts\uvicorn api:app --port 8000
```
Leave this window open. Stop it with `Ctrl+C`.

### 3.2 Easiest way: the browser
Open **http://localhost:8000/docs**. It's an interactive page where you can:
1. Expand **POST /ingest** → *Try it out* → choose files → *Execute*
2. Expand **POST /query** → *Try it out* → edit the JSON → *Execute*

### 3.3 Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/ingest` | Upload one or more files (PDF, DOCX, TXT, MD; max 25 MB each) |
| `POST` | `/query` | Ask a question → answer + cited sources |
| `GET` | `/stats` | Which documents are indexed and how many chunks |
| `GET` | `/health` | `{"status": "ok"}` if the server is up |

### 3.4 From PowerShell
```powershell
# upload (curl.exe, not the PowerShell alias)
curl.exe -F "files=@D:\docs\report.pdf" -F "files=@D:\docs\policy.docx" http://localhost:8000/ingest

# ask
curl.exe -X POST http://localhost:8000/query -H "Content-Type: application/json" `
  -d '{\"query\": \"What is the budget?\", \"k\": 4}'

# or with native PowerShell
Invoke-RestMethod http://localhost:8000/query -Method Post -ContentType "application/json" `
  -Body (@{ query = "What is the budget?"; k = 4 } | ConvertTo-Json)
```

### 3.5 From Python
```python
import httpx

with open("report.pdf", "rb") as f:
    print(httpx.post("http://localhost:8000/ingest", files={"files": f}).json())

r = httpx.post("http://localhost:8000/query", json={"query": "What is the budget?", "k": 4}, timeout=120)
print(r.json()["answer"])
```

### 3.6 Reading the answer
```json
{
  "answer": "The inverter needs at least 30 cm clearance on every side [1].",
  "sources": [
    {"id": 1, "file": "solar_handbook.pdf", "page": 3, "chunk_id": "solar_handbook.pdf#2",
     "score": 0.6872, "snippet": "Inverter The HX system uses the Volta S6..."}
  ],
  "latency_ms": 3799
}
```
- **`[1]`** in the answer refers to the source with `"id": 1`.
- **`score`**: similarity from -1 to 1; higher means more relevant. Above ~0.5 is usually a good match.
- **`page`** is `null` for DOCX/TXT/MD, which have no pages.
- If the documents don't contain the answer, it replies **"I don't know based on the provided documents."** instead of guessing.
- **`k`** (1–20, default 4) is how many chunks are retrieved as context.

### 3.7 Errors you may see

| HTTP | Meaning | Fix |
|---|---|---|
| 413 | File larger than `MAX_UPLOAD_MB` | Split it, or raise the limit in `.env` |
| 415 | Unsupported file type | Use PDF, DOCX, TXT or MD |
| 422 | Corrupt file, empty or scanned file, or invalid request | Check the file; scanned PDFs need OCR |
| 503 | Ollama isn't reachable | Start Ollama (`ollama serve`) |

### 3.8 Updating and deleting documents
- **Update:** upload a file with the **same name**. Its old chunks are replaced.
- **Start fresh:** stop the server and delete `rag-docs-qa\index\` and `rag-docs-qa\data\`.

### 3.9 Measure quality (evaluation)
```powershell
.\.venv\Scripts\python eval\run_eval.py --k 3
```
Prints per question whether the right document was retrieved (HIT/MISS) and a 1–5 relevance grade from the model, then:
```
Hit-Rate@3: 100.00%   Answer Relevance: 4.00/5 (75%)
```
Add your own cases to `eval\eval_set.json` (and their documents to `eval\docs\`) to test on your data. Use it to compare models: change `LLM_MODEL` and rerun.

### 3.10 Docker (optional)
Ollama keeps running on your PC; the container connects to it.
```powershell
docker build -t rag-qa .
docker run -p 8000:8000 -v ${PWD}/index:/app/index -v ${PWD}/data:/app/data rag-qa
```
On Linux, add `--add-host=host.docker.internal:host-gateway`.

---

## 4. Configuration (`.env`)

All settings are optional. Put them in a `.env` file inside the project folder, or set them as environment variables.

**Both projects**

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama runs |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model (**re-index everything if you change it**) |

**Agent Toolkit only**

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_ORDER` | `gemini,ollama` | Models to try, in order |
| `OLLAMA_MODEL` | `qwen2.5-coder:3b` | Ollama answer model |
| `GEMINI_CMD` / `GEMINI_MODEL` | `gemini` / *(CLI default)* | Gemini CLI command and model |
| `MODEL_TIMEOUT` / `MODEL_RETRIES` | `120` / `1` | Seconds per call / extra attempts per model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunk length and overlap (characters) |
| `KB_DIR` | `kb` (in project) | Where the knowledge base is stored |
| `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT` | `stdio`, `127.0.0.1`, `8765` | MCP transport settings |

**RAG Service only**

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `qwen2.5-coder:3b` | Ollama answer model |
| `INDEX_DIR` / `DATA_DIR` | `index` / `data` | Index location / where uploaded files are kept |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunk length and overlap (characters) |
| `MAX_UPLOAD_MB` | `25` | Upload size limit per file |

---

## 5. Run the tests
No models needed; they use fakes.
```powershell
cd D:\test\project\local-agent-toolkit; .\.venv\Scripts\python test_toolkit.py
cd D:\test\project\rag-docs-qa;         .\.venv\Scripts\python -m pytest -q
```

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Failed: all models failed -> ...` | Read the reasons in the message. Usually Ollama isn't running (`ollama serve`) or the model isn't pulled (`ollama list`). |
| `'gemini' not found on PATH` warnings | Harmless; set `MODEL_ORDER=ollama` to silence them. |
| "Knowledge base is empty" | Index first: `orchestrator.py "index the pdfs in <folder>"` |
| Answer says it doesn't know, but the info is in the file | Small-model limitation. Try a bigger `OLLAMA_MODEL`/`LLM_MODEL`, raise `k`, or rephrase the question. |
| Vague summaries / agent forgets details | Same: a 7B+ model is noticeably better than the default 3B. |
| Weird results after changing `EMBED_MODEL` | Old vectors are incompatible. Delete `kb\` / `index\` and re-index. |
| Port 8000 already in use | `uvicorn api:app --port 8001`, or stop the old server: `Get-NetTCPConnection -LocalPort 8000 -State Listen \| % { Stop-Process -Id $_.OwningProcess }` |
