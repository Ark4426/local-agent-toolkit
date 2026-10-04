# Project 2: Local AI Agent Toolkit – Implementation Plan

## 1. Overview

Build a local-first AI agent toolkit that orchestrates specialized LLM agents to route and solve tasks. The system supports **Gemini CLI as primary** with **Ollama fallback**, includes a **FAISS vector search over PDFs** using **`nomic-embed-text`** embeddings, and exposes the knowledge-base capabilities as **MCP (Model Context Protocol) tools** for agent/tool calling. The proposed architecture is a hybrid split: a **Python MCP-RAG server** (FAISS, embeddings, PDF ingestion, MCP tools) and an **Orchestrator** (agent registry, routing, model abstraction, tool-use loop) which can be implemented in **Node.js** (as requested: “Node.js/pythobn agent orchestration”) or Python. This plan covers both components, interfaces, fallback logic, testing, and optional Dockerization.

## 2. Goals

- Agent orchestration layer that routes tasks to specialized agents (Research, Summarizer, Q&A, Document-Extractor).
- Model routing with **Gemini CLI (primary)** and **Ollama (fallback)** with availability checks, timeouts, retries, and clear failure reasons.
- Local PDF knowledge base: ingest PDFs → chunk → embed with `nomic-embed-text` → FAISS vector search, persisted to disk.
- Expose KB/search/ingestion as **MCP tools** (`search_pdfs`, `index_pdfs`, `list_pdfs`) via a FastMCP server for agent/tool calling.
- Support agent **tool-use loops** (agents can call MCP tools, observe results, iterate until completion).
- Keep fully local-first by default; treat Gemini CLI availability as optional/primary (graceful fallback).
- Provide unit/integration tests and a minimal smoke-testable workflow.
- Modular, configurable via `.env`, with clear interfaces and logging.

## 3. Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **MCP-RAG (KB + MCP)** | Python 3.11+ | FAISS, embeddings, PDF loading, MCP server |
| **MCP Framework** | [`FastMCP`](https://github.com/modelcontextprotocol/python-sdk) (MCP Python SDK) | Expose tools/resources via MCP (stdio or HTTP transport) |
| **Orchestrator** | **Node.js (TypeScript)** preferred per “Node.js/pythobn”, or **Python** (LangGraph/AutoGen) | Agent registry, routing, model clients, tool-use loop, CLI/API |
| **Agent Framework (optional)** | [`LangGraph.js`](https://www.langchain.com/langgraph-js) / [`LangChain.js`](https://js.langchain.com/) (Node) or [`LangGraph`](https://www.langchain.com/langgraph) (Python) | State machines, tool calling, agent loops |
| **PDF Loaders** | [`PyMuPDF`](https://pymupdf.readthedocs.io/) (`fitz`) | Extract text + page numbers from PDFs |
| **Chunking** | `RecursiveCharacterTextSplitter` (`langchain-text-splitters`) | Overlap-aware chunking |
| **Embeddings** | [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) via [Ollama](https://ollama.com/) (recommended) or [HuggingFace](https://huggingface.co/nomic-ai) (`nomic-ai/nomic-embed-text-v2*`) | Specified embedding model |
| **Vector Store** | [FAISS](https://faiss.ai/) (`faiss-cpu`) | Local similarity search |
| **Primary LLM** | [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Primary model via non-interactive CLI invocation (subprocess) |
| **Fallback LLM** | [Ollama](https://ollama.com/) (e.g. `qwen2.5-7b-instruct`, `llama-3.1-8b-instruct`, `mistral-nemo`, `deepseek-r1:8b-q5_K_M`) | Local fallback |
| **MCP Client (Orchestrator)** | [`@modelcontextprotocol/sdk`](https://github.com/modelcontextprotocol/typescript-sdk) (Node) or MCP Python client | Call MCP tools (`search_pdfs`, `index_pdfs`, `list_pdfs`) |
| **Config** | `.env` (`dotenv`) | Environment configuration |
| **Logging** | `pino` (Node) or `structlog/logging` (Python) | Structured logs |
| **Testing** | [`Vitest`](https://vitest.dev/) / `Jest` (Node) or [`pytest`](https://docs.pytest.org/) (Python) | Unit/integration tests |

## 4. Architecture

Recommended hybrid architecture (matches “Node.js/pythobn agent orchestration” while keeping FAISS/MCP clean in Python).

```text
User/Task
  ↓
Orchestrator (Node.js/TypeScript)  ← Agent Registry, Router, Model Router (Gemini CLI + Ollama)
  ├─ Model Clients: gemini-cli.ts, ollama.ts (OpenAI-compatible)
  ├─ Agent Registry: Research, Summarizer, Q&A, Document-Extractor
  ├─ Tool-Use Loop: plan → tool_call(MCP) → observe → iterate
  └─ MCP Client → stdio/HTTP → MCP-RAG (Python/FastMCP)
                                 ├─ Tools: index_pdfs, search_pdfs, list_pdfs
                                 ├─ PDF Loaders + Chunker
                                 ├─ Embeddings (nomic-embed-text, Ollama/HF)
                                 └─ FAISS KB (./kb/index.faiss + metadata.json)
```

### 4.1 Components

| Component | Type | Responsibilities |
|---|---|---|
| **`mcp-rag/`** | Python (FastMCP) | PDF ingestion, chunking, nomic-embed-text embeddings, FAISS build/load/persist, MCP tool exposure (`index_pdfs`, `search_pdfs`, `list_pdfs`). Stateless, reusable. |
| **`orchestrator/`** | Node.js/TS (recommended) | Task intake, intent classification/routing, specialized agent definitions, model abstraction + primary/fallback, MCP client invocations, tool-use loop, memory/session state, CLI entrypoint. |
| **Agent Registry** | Orchestrator | Maps task type/intent → agent spec (role, system prompt, allowed tools, max tool calls, model preference, temperature). |
| **Router** | Orchestrator | Selects best specialized agent (rule-based keyword matching to start; upgradeable to LLM classifier). |
| **Model Router** | Orchestrator | `ModelClient` interface with `GeminiCliClient` (primary) and `OllamaClient` (fallback). Implements availability check, try-primary → on error/timeout/unavailable → fallback with reason logging. |
| **Tool-Use Loop** | Orchestrator | ReAct-style: Thought → Tool Call(s) (MCP) → Tool Result → Repeat until answer/stop or `max_tool_calls`. |

### 4.2 Transport (MCP)

| Transport | Use Case | Notes |
|---|---|---|
| **stdio** | Local-only, simplest, secure | Default for single-machine local toolkit. Orchestrator spawns `mcp-rag` as subprocess or connects via stdio. |
| **HTTP (Streamable/HTTP+SSE)** | Split processes/containers or remote-local | Optional. Enable via `MCP_TRANSPORT=http` + `MCP_HOST/PORT`. |