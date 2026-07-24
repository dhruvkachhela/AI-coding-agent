# Codebase RAG & Automated Repair Agent with Dual-Mode Grounding & Execution Verifier

A production-grade, dual-mode ReAct & Code Repair Agent built from scratch using **LangGraph**, **Tree-sitter**, **ChromaDB**, **BM25**, **Pytest Sandbox Subprocesses**, and a multi-tiered **Model Allocation Pipeline**. The agent dynamically navigates repository codebases using hybrid retrieval (dense embedding + field-weighted BM25 with Reciprocal Rank Fusion) and enforces strict zero-hallucination & verified bug resolution guarantees.

The system was evaluated and tested end-to-end against **VibeCheck Scan / VibeSec Pipeline** (a 9-layer AI-driven static application security testing scanner containing cross-file imports, database schemas, and AI-triage engines).

---

## 1. Motivation & Technical Focus

While traditional RAG pipelines rely on single-shot document retrieval and direct LLM generation, codebase QA and automated bug repair present three unique challenges:
1. **Structural Blind Spots**: High-level architectural flows (e.g. entry points, call-graph chains) cannot be captured by single-turn semantic search alone.
2. **LLM Hallucinations**: Standard ReAct agents often synthesize plausibly sounding but unverified claims (e.g., hallucinating API endpoint paths, missing wrapper layers, or misrepresenting class inheritance).
3. **Unverified Code Fixes**: AI-generated code patches frequently introduce AST syntax errors, invalid imports, or breaking regressions if returned without execution testing.

This project resolves these challenges by implementing a **Dual-Mode LangGraph State Machine**:
- **Category 1 (QA Mode)**: Paired with an automated **Grounding Critic LLM** that audits final answers against raw retrieved code observations before returning them to the user.
- **Category 2 (Bug Fix Mode)**: Paired with an **Execution Verifier Engine** that validates AST syntax, dynamically generates pytest reproduction scripts, and executes pre/post fix tests inside an isolated sandbox subprocess.

---

## 2. System Architecture

```
                                    +------------------------------------+
                                    |        Input User Question         |
                                    +-------------------+----------------+
                                                        |
                                                        v
                                    +-------------------+----------------+
                                    |     Intent Classifier Node         |
                                    |   (Llama-3.1-8B | Temp 0.0)        |
                                    +-------------------+----------------+
                                                        |
                                       [QA vs FIX_PROPOSAL Classification]
                                                        |
                                                        v
                                    +-------------------+----------------+
     +----------------------------->|          Reasoning Node            |<----------------+
     |                              |   (Llama-3.1-8B | Temp 0.1)        |                 |
     |                              +-------------------+----------------+                 |
     |                                                  |                                  |
     |                                    [Conditional Edge Decision]                      |
     |                                                  |                                  |
     |                                  Is there a Final Answer?                           |
     |                                    /                     \                          |
     |                                  Yes                      No (Search Action)        |
     |                                  /                         \                        |
     |                                 v                           v                       |
     |                        Category 1 or 2?             +-------+--------+              |
     |                        /              \             |   Tool Node    |              |
     |                       /                \            | (Hybrid Search)|              |
     |                 Category 1          Category 2      +-------+--------+              |
     |                     /                    \                  |                       |
     |                    v                      v                 v                       |
     |        +-------+-------+          +-------+-------+ +-------+--------+              |
     |        | Verifier Node |          | Fix Proposal  | | Dense  + BM25  |              |
     |        | (Grounding)   |          | (Codestral)   | |   RRF Fusion   |              |
     |        +-------+-------+          +-------+-------+ +-------+--------+              |
     |                |                          |                 |                       |
     |         [Grounding Check]                 v                 |                       |
     |          /           \            +-------+-------+         |                       |
     |     Supported    Unsupported      | Execution     |         |                       |
     |       /                 \         | Verifier      |         |                       |
     |      v                   v        +-------+-------+         |                       |
     |   +-----+        (Self-Correction)        |                 |                       |
     |   | END |                 \               v                 |                       |
     |   +-----+                  +------------->+                 |                       |
     |                                           |                 |                       |
     +-------------------------------------------+-----------------+-----------------------+
                                        Appends Observation to State
```

### 2.1 AST-Based Structural Chunking (Method-Level)
Instead of dividing code into arbitrary character windows (which breaks function syntax and context boundaries), the repository chunker uses **Tree-sitter** to parse Python code into an Abstract Syntax Tree (AST).
* **The Granularity Refactoring**: Initially, entire classes were parsed as single chunks. However, this produced massive outliers: the main `CodeIndexer` class spanned 1,900+ lines (22,956 tokens), diluting vector embeddings.
* **Method-Level Extraction**: The chunker recursively descends into class structures, extracting **individual methods** as separate chunks. To preserve class context, each method chunk is prefixed with class-level metadata:
  ```text
  File: scanner/layer7_validator.py
  Class: ValidationEngine
  Method: _tier3_joern
  Type: method
  ```
* **Impact**: Maximum chunk size dropped from **22,956 to 7,005 tokens**, and average chunk size decreased from **707.18 to 422.77 tokens**.

### 2.2 Embedding Model Selection
We evaluated multiple embedding models against the codebase chunk distribution:
1. `sentence-transformers/all-MiniLM-L6-v2`: 256-token context window. Truncates over 60% of code chunks.
2. `nvidia/nv-embedcode-v1`: 512-token context window. Truncates the 95th percentile of chunks (1,553.40 tokens).
3. `nvidia/llama-nemotron-embed-1b-v2`: 8,192-token context window.

**Decision**: We selected `llama-nemotron-embed-1b-v2` to guarantee zero truncation across 100% of the repository's code chunks. We use `input_type="passage"` during indexing and `input_type="query"` during query retrieval.

### 2.3 Hybrid Search & RRF Fusion
To combine deep semantic vector matching with exact symbol/variable lookups:
* **Dense Stream**: Local **ChromaDB** vector store using Cosine Similarity.
* **Sparse (BM25) Stream**: Dual-field `rank_bm25` indexes:
  * `metadata_index` (File path, Class name, Method name) — Weight: **3.0**
  * `body_index` (Raw source code) — Weight: **1.0**
* **Fusion**: Top 20 candidates from both streams are fused using **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF\_Score}(d) = \frac{1}{k + \text{Rank}_{\text{dense}}(d)} + \frac{1}{k + \text{Rank}_{\text{BM25}}(d)}$$

---

## 3. Dual-Mode Verification Architecture

### Mode A: Grounding Critic & Verifier (`verifier_node`)
For Category 1 (QA) queries, the agent routes candidate answers to a strict Grounding Critic:
- **Unbiased Context**: The verifier receives **ONLY** the proposed `Final Answer` and consolidated `Retrieved Observations` across search turns (intermediate thoughts and queries are excluded).
- **Grounding Audit**: The critic (`meta/llama-3.1-8b-instruct` at `temperature=0.0`) evaluates if every claim is explicitly backed by retrieved code.
- **Self-Correction Loop**: If unsupported, it injects feedback back to `reasoning_node` (max 3 attempts).

### Mode B: Fix Proposal & Sandboxed Execution Verifier (`execution_verifier_node`)
For Category 2 (Bug Fix Proposal) queries:
1. **Fix Synthesis (`fix_proposal_node`)**: `Codestral-22B` generates structured diagnosis, relative target file path, original code snippet, and fixed replacement code snippet.
2. **AST Syntax Check (`ast.parse`)**: Validates replacement code syntax prior to execution.
3. **Dynamic Unit Test Generation**: LLM dynamically writes a standalone `pytest`/`unittest` reproduction script.
4. **Sandboxed Subprocess Execution**:
   - Creates an isolated temp directory (`tempfile.mkdtemp(prefix="agent_sandbox_")`).
   - Writes test code and runs `subprocess.run([sys.executable, "-m", "pytest", ...], timeout=15)`.
   - Asserts pre-fix failure and post-fix success with zero regressions before returning final report.

---

## 4. Per-Node Model Specialization & Latency Optimization

To eliminate queueing bottlenecks on shared API endpoints and maximize speed:

| Node Name | Configured Model | Temperature | Max Tokens | Rationale & Architectural Decision |
| :--- | :--- | :--- | :--- | :--- |
| **`intent_classifier`** | `meta/llama-3.1-8b-instruct` | `0.0` | `20` | **Instant Routing**: Binary `QA` vs `FIX_PROPOSAL` decision runs in **~0.2s** (eliminates 10s heavy model bottleneck). |
| **`reasoning_node`** | `meta/llama-3.1-8b-instruct` | `0.1` | `512` | **Ultra-Fast ReAct Loop**: 8B model eliminates server queue delays (**~1.2s per turn** vs 43s on 70B models). Capped at `512` tokens. |
| **`verifier_node`** | `meta/llama-3.1-8b-instruct` | `0.0` | `1024` | **Deterministic Fact-Checking**: Strict anti-hallucination grounding audit (~0.8s). |
| **`fix_proposal_node`** | `mistralai/codestral-22b-instruct-v0.1` | `0.2` | `3072` | **Specialized Code Synthesis**: 22B code model ensures high AST diff precision and prevents code truncation. |
| **`execution_verifier_node`**| `mistralai/codestral-22b-instruct-v0.1` | `0.1` | `2048` | **Precise Unit Test Generation**: Generates clean, executable `pytest` scripts. |

---

## 5. Built-in Latency Benchmarking & Colored UI Output

The agent automatically tracks execution time per node in `AgentState["node_latencies"]` and outputs a benchmark report at the end of every run:

```text
==========================================================================================
                 DUAL-MODE AGENT LATENCY & PERFORMANCE BENCHMARK REPORT
==========================================================================================
Node Name            | Model ID                       | Calls  | Total (s)  | Avg (s)   | % Total
------------------------------------------------------------------------------------------
intent_classifier    | meta/llama-3.1-8b-instruct     | 1      | 0.214      | 0.214     | 3.8%   
reasoning            | meta/llama-3.1-8b-instruct     | 3      | 3.842      | 1.280     | 68.4%  
tool                 | hybrid_search (AST+BM25+DB)    | 2      | 0.112      | 0.056     | 2.0%   
verifier             | meta/llama-3.1-8b-instruct     | 1      | 1.450      | 1.450     | 25.8%  
------------------------------------------------------------------------------------------
Total Agent Execution Latency: 5.618 seconds
==========================================================================================
```

### Color Coding Scheme:
- **Final Verified Agent Output**: Bold Bright Green (`\033[1;92m`)
- **Reasoning & Tool Logs**: Bold Cyan (`\033[1;96m`)
- **Verifier Critic Logs**: Bold Yellow (`\033[1;93m`)
- **Benchmark Summary Header**: Bold Magenta (`\033[1;95m`)

---

## 6. Technology Stack

- **Agent Orchestration**: `langgraph` (v1.1.9)
- **AST Parser**: `tree-sitter` (v0.26.0) & `tree-sitter-python` (v0.25.0)
- **Vector Database**: `chromadb` (v1.5.9)
- **Sparse Retrieval**: `rank_bm25` (v0.2.2)
- **Sandbox Subprocess Engine**: `subprocess` + `pytest` (v8.x) + `tempfile`
- **LLM Provider**: **NVIDIA NIM API** (`integrate.api.nvidia.com/v1`) via `openai` Python SDK
- **Embedding Model**: `nvidia/llama-nemotron-embed-1b-v2` via NVIDIA NIM API

---

## 7. Setup & Execution

### Prerequisites
Set your NVIDIA NIM API key:
```bash
export NVIDIA_API_KEY="nvapi-..."
```

### Install Dependencies
```bash
pip install --upgrade opentelemetry-api opentelemetry-sdk chromadb tree-sitter tree-sitter-python openai tqdm rank_bm25 langgraph pytest
```

### Running in Jupyter / Kaggle
Open `agent.ipynb` and run the cells sequentially:
- **Cell 1–6**: Dependencies, AST Chunker, Vector Embedder, and Hybrid Search Engine functions.
- **Cell 10**: `build_agent_graph()` definition with Dual-Mode State Machine, Latency Engine, and Verifiers.
- **Cell 11**: Execution runner testing sample QA and Bug Fix queries with colored outputs & benchmark reporting.
