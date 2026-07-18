# Codebase RAG Agent

**Status: Planning / Not Yet Built.** This README documents the intended scope and architecture before implementation begins. Nothing described below has been built, run, or evaluated yet — this is a design document, not a results writeup.

---

## 1. What This Project Is

An agent that can answer non-trivial questions about a real, existing codebase — "where is X handled," "what calls this function," "what would break if I changed this" — and optionally propose a code fix, by combining:

- **Retrieval-Augmented Generation (RAG)** over the target codebase, so the agent grounds its answers in actual code rather than guessing from pretraining knowledge.
- **A ReAct-style reasoning loop** (Thought → Action → Observation), so the agent can decide what to retrieve, evaluate whether it has enough context, and iterate rather than answering in one shot.

This is deliberately scoped as the same category of tool as Cursor, Claude Code, and Devin-style coding agents — not a novel product idea, but a from-scratch build of the core mechanism behind that category, done to genuinely understand it rather than to wrap an existing framework.

---

## 2. Why This Project (Not Another Fine-Tune)

Prior projects (`qlora-nl2sql`, `nl2sql-lora`, `flatland_gpt`) covered transformer internals, quantization, and fine-tuning in depth. This project deliberately targets a different, currently-missing skill area: **agentic reasoning and retrieval systems**, which is a distinct and increasingly central part of AI engineering work in 2026.

---

## 3. Architecture (Planned)

### 3.1 Chunking
- Code is chunked by **structural boundary (function/class), not fixed character count** — using Tree-sitter for AST-aware parsing (reusing prior hands-on experience from the VibeSec project).
- Rationale: naive fixed-size chunking frequently splits a function mid-body, destroying retrieval quality. Structural chunking keeps semantically complete units intact.

### 3.2 Embedding + Storage
- Each code chunk is embedded into a dense vector.
- Vectors are stored in **Chroma** (local, free, fast to iterate on).
- Retrieval: given a natural-language question, embed the question the same way, query Chroma for the top-k nearest chunks by similarity.

### 3.3 Agent Loop
- Built in **LangGraph**, implementing the ReAct pattern: the agent reasons about what it needs, calls the retrieval tool, observes the result, and either answers or issues another retrieval/reasoning step.
- Retrieval is exposed as a **tool call the agent invokes**, not a fixed pre-generation step — this is what distinguishes it from "naive RAG."

### 3.4 Target Repository
- TBD: a real, moderate-size open-source repository (not a toy codebase), chosen so retrieval and reasoning are tested against genuine cross-file complexity.

---

## 4. Build Sequence

Built incrementally, each stage verified before the next is added — same discipline as the from-scratch QLoRA build (verify the primitive, then build on it):

1. **Chunking + embedding + Chroma retrieval**, verified in isolation (does it return the right chunk for a known question) before any agent logic is added.
2. **Single-tool ReAct loop** (retrieval only) in LangGraph — agent can answer questions grounded in retrieved code.
3. **Fix-proposal capability** — agent can propose a code change for a described issue, using retrieved context.
4. **(Stretch, later phase) Second verifier agent** — an independent agent checks proposed fixes against the test suite, looping with the first agent on failure. Treated as a natural extension of a working single-agent system, not attempted from scratch.

---

## 5. Explicitly Out of Scope (For Now)

Named up front so scope doesn't silently creep:

- **Adaptive RAG (query routing)** and **reranking (cross-encoder second stage)** — real 2026 production techniques, planned as a follow-up layer once the base retrieval+agent loop is verified working, not part of the initial build.
- **GraphRAG** — evaluated and deliberately excluded; current evidence suggests it adds latency without consistent benefit on simple lookups, which don't justify the complexity for this project's scope.
- Scaling to millions of documents / production-grade ANN tuning (HNSW/IVF internals) — relevant for interview-level discussion, not required for this project's actual scale (one target repo).

---

## 6. What Will Be Documented Once Built

In keeping with prior projects' reporting standard: real numbers, real failure modes, no inflated claims. This section will be filled in with actual results, including:
- What retrieval quality looked like before/after chunking strategy decisions.
- Concrete cases where the agent's reasoning loop failed or looped incorrectly.
- Honest evaluation of proposed-fix quality, not just "it worked once."
