"""Orchestrator: routes a task to a specialized agent, runs a JSON tool-use loop over MCP tools,
with Gemini CLI as primary model and Ollama as fallback.

    python orchestrator.py "summarize the indexed papers" [--agent summarizer]
"""
import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
log = logging.getLogger("orchestrator")

MODEL_ORDER = os.getenv("MODEL_ORDER", "gemini,ollama").split(",")
GEMINI_CMD = os.getenv("GEMINI_CMD", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
TIMEOUT = float(os.getenv("MODEL_TIMEOUT", 120))
RETRIES = int(os.getenv("MODEL_RETRIES", 1))

AGENTS = {
    "research": {
        "system": "You are a Research agent. Run several searches with different phrasings, cross-check sources, "
                  "and synthesize findings. Cite every claim as (file, page).",
        "tools": ["search_pdfs", "list_pdfs"], "max_tool_calls": 6, "temperature": 0.4,
        "keywords": ["research", "investigate", "compare", "explore", "find out", "analyze"],
    },
    "summarizer": {
        "system": "You are a Summarizer agent. Retrieve the relevant passages and write a concise, structured "
                  "summary with key points. Cite (file, page).",
        "tools": ["search_pdfs", "list_pdfs"], "max_tool_calls": 3, "temperature": 0.3,
        "keywords": ["summarize", "summary", "summarise", "tl;dr", "overview", "gist"],
    },
    "extractor": {
        "system": "You are a Document-Extractor agent. Ingest documents when asked, and extract structured data "
                  "(names, dates, numbers, tables) as JSON, citing (file, page).",
        "tools": ["index_pdfs", "search_pdfs", "list_pdfs"], "max_tool_calls": 5, "temperature": 0.0,
        "keywords": ["extract", "index", "ingest", "add pdf", "list", "table", "fields", "json"],
    },
    "qa": {
        "system": "You are a Q&A agent. Answer precisely from the knowledge base and cite (file, page). "
                  "If the answer is not in the documents, say so.",
        "tools": ["search_pdfs"], "max_tool_calls": 3, "temperature": 0.1,
        "keywords": [],  # default agent
    },
}

PROTOCOL = """You can use these tools:
{tools}

Reply with ONLY one JSON object, nothing else:
  {{"tool": "<tool name>", "args": {{...}}}}   to call a tool, or
  {{"answer": "<final answer>"}}               when you are done."""


def route(task):
    """Rule-based router. ponytail: keyword scoring; swap for an LLM classifier if routing gets fuzzy."""
    t = task.lower()
    best = max(AGENTS, key=lambda a: sum(k in t for k in AGENTS[a]["keywords"]))
    return best if any(k in t for k in AGENTS[best]["keywords"]) else "qa"


class ModelError(Exception):
    pass


class ModelUnavailable(ModelError):
    """Permanent failure (e.g. CLI not installed): skip retries."""


def call_gemini(prompt, temperature):
    exe = shutil.which(GEMINI_CMD)
    if not exe:
        raise ModelUnavailable(f"'{GEMINI_CMD}' not found on PATH")
    # Prompt goes through stdin: multi-line argv is unreliable through .cmd shims on Windows.
    args = [exe, "-p", "Follow the instructions above."] + (["-m", GEMINI_MODEL] if GEMINI_MODEL else [])
    try:
        r = subprocess.run(args, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        raise ModelError(f"timed out after {TIMEOUT}s")
    if r.returncode != 0 or not r.stdout.strip():
        raise ModelError(f"exit {r.returncode}: {r.stderr.strip()[-300:]}")
    return r.stdout.strip()


def call_ollama(prompt, temperature):
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate", timeout=TIMEOUT, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": temperature}})
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise ModelError(f"{type(e).__name__}: {e}")
    return r.json()["response"].strip()


MODELS = {"gemini": call_gemini, "ollama": call_ollama}


def complete(prompt, temperature=0.2):
    """Try each model in MODEL_ORDER (with retries); first success wins, else raise with every reason."""
    errors = []
    for name in MODEL_ORDER:
        for attempt in range(1, RETRIES + 2):
            try:
                out = MODELS[name.strip()](prompt, temperature)
                log.info("model=%s ok", name)
                return out
            except ModelError as e:
                log.warning("model=%s attempt=%d failed: %s", name, attempt, e)
                errors.append(f"{name}#{attempt}: {e}")
                if isinstance(e, ModelUnavailable):
                    break  # retrying can't fix a missing binary
    raise ModelError("all models failed -> " + " | ".join(errors))


def parse_action(text):
    """First JSON object in the reply containing 'tool' or 'answer'. Non-JSON reply = final answer."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and ("tool" in obj or "answer" in obj):
                return obj
    return {"answer": text}


async def agent_loop(session, task, agent):
    spec = AGENTS[agent]
    tools = [t for t in (await session.list_tools()).tools if t.name in spec["tools"]]
    # Plain signatures: small models copy raw JSON schemas back as argument values.
    def sig(t):
        params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in t.inputSchema.get("properties", {}).items())
        return f"- {t.name}({params}) - {(t.description or '').splitlines()[0]}"
    tool_doc = "\n".join(map(sig, tools))
    transcript = f"{spec['system']}\n\n{PROTOCOL.format(tools=tool_doc)}\n\nTask: {task}\n"

    for step in range(spec["max_tool_calls"] + 1):
        if step == spec["max_tool_calls"]:
            transcript += "\nTool budget exhausted. Reply now with the final answer JSON.\n"
        action = parse_action(await asyncio.to_thread(complete, transcript, spec["temperature"]))
        if "answer" in action and step == 0:  # small models skip retrieval and hallucinate; force one tool call
            transcript += "\nYou have not used any tool yet. Call a tool first; do not answer from memory.\n"
            continue
        if "answer" in action or step == spec["max_tool_calls"]:
            return str(action.get("answer", action))
        name, args = action["tool"], action.get("args") or {}
        if name not in spec["tools"]:
            result = f"ERROR: tool '{name}' is not available to this agent."
        else:
            log.info("tool=%s args=%s", name, args)
            res = await session.call_tool(name, args)
            result = ("ERROR: " if res.isError else "") + "\n".join(getattr(c, "text", "") for c in res.content)
        transcript += f"\nYou called: {json.dumps(action)}\nResult:\n{result[:6000]}\n"


async def run(task, agent=None):
    agent = agent or route(task)
    log.info("agent=%s", agent)
    if os.getenv("MCP_TRANSPORT") == "http":
        client = streamablehttp_client(f"http://{os.getenv('MCP_HOST', '127.0.0.1')}:{os.getenv('MCP_PORT', 8765)}/mcp")
    else:
        client = stdio_client(StdioServerParameters(command=sys.executable, args=[str(HERE / "mcp_rag.py")],
                                                    env=dict(os.environ)))
    async with client as streams, ClientSession(streams[0], streams[1]) as s:
        await s.initialize()
        try:
            return await agent_loop(s, task, agent)
        except ModelError as e:  # re-raise outside the MCP task group, which would wrap it in an ExceptionGroup
            err = e
    raise err


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Local AI Agent Toolkit")
    ap.add_argument("task")
    ap.add_argument("--agent", choices=AGENTS)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows pipes default to cp1252 and crash on non-Latin model output
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, stream=sys.stderr,
                        format="%(levelname)s %(name)s %(message)s")
    try:
        print(asyncio.run(run(a.task, a.agent)))
    except ModelError as e:
        sys.exit(f"Failed: {e}")
