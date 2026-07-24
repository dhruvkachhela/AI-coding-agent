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
                                                        v
     +--------------------------------------------------+---------------------------------------+
     |                                                  |                                       |
     |                              +-------------------+----------------+                      |
     |                              |          Reasoning Node            |<-----------------+   |
     |                              |   (Llama-3.1-8B | Temp 0.1)        |                  |   |
     |                              +-------------------+----------------+                  |   |
     |                                                  |                                   |   |
     |                                    [Conditional Edge Decision]                       |   |
     |                                                  |                                   |   |
     |                                  Is there a Final Answer?                            |   |
     |                                    /                     \                           |   |
     |                                  Yes                      No (Search Action)         |   |
     |                                  /                         \                         |   |
     |                                 v                           v                        |   |
     |                        Category 1 or 2?             +-------+--------+               |   |
     |                        /              \             |   Tool Node    |               |   |
     |                       /                \            | (Hybrid Search:|               |   |
     |                 Category 1          Category 2      |  Dense + BM25  |               |   |
     |                     /                    \          |  RRF Fusion)   |               |   |
     |                    v                      v         +-------+--------+               |   |
     |        +-------+-------+          +-------+-------+         |                        |   |
     |        | Verifier Node |          | Fix Proposal  |         | (Appends Observation)  |   |
     |        | (Grounding)   |          | (Codestral)   |         +------------------------+   |
     |        +-------+-------+          +-------+-------+                                      |
     |                |                          |                                              |
     |         [Grounding Check]                 v                                              |
     |          /           \            +-------+-------+                                      |
     |     Supported     Unsupported     | Execution     |                                      |
     |       /               \           | Verifier      |                                      |
     |      v                 v          +-------+-------+                                      |
     |   +-----+      (Self-Correction:          |                                              |
     |   | END |       QA Re-Reason)             v                                              |
     |   +-----+              |          [Test Status Check]                                    |
     |                        |          /                 \                                    |
     |                        |      Passed               Failed                                |
     |                        |        /                     \                                  |
     |                        |       v                       v                                 |
     |                        |    +-----+            (Self-Correction:                         |
     |                        |    | END |             Fix Patch Retry)                         |
     |                        |    +-----+                    |                                 |
     |                        |                               |                                 |
     +------------------------+-------------------------------+---------------------------------+
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
| **`fix_proposal_node`** | `mistralai/codestral-22b-instruct-v0.1` | `0.1` | `3072` | **Specialized Code Synthesis**: 22B code model ensures high AST diff precision and prevents code truncation. |
| **`execution_verifier_node`**| `mistralai/codestral-22b-instruct-v0.1` | `0.0` | `2048` | **Precise Unit Test Generation**: Generates clean, deterministic `pytest` scripts. |

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

## 6. Empirical Research, Ablation Studies & Failures Analysis

To systematically determine the optimal model allocation per node, we conducted four empirical research trials across different LLM parameter classes and providers on NVIDIA NIM API endpoints. Below is a rigorous breakdown of our findings, failures, and architectural insights.

---

### 6.1 Model Provider & Endpoint Availability Analysis (MiniMax 3 & Kimi K2.6)

During our investigation into third-party foundation models:
* **MiniMax 3 (`minimax/minimax-text-01`)** and **Moonshot Kimi K2.6 (`moonshotai/kimi-k1.5`)**: Returned `404 Page Not Found` on NVIDIA NIM API (`https://integrate.api.nvidia.com/v1`). Proprietary models like MiniMax and Kimi are hosted exclusively on their native provider APIs (`api.minimax.chat` and `api.moonshot.cn`).
* **NVIDIA NIM Models**: NVIDIA NIM endpoints prioritize open-weights models (`Llama 3.1`, `Llama 3.3`, `GLM-5.2`, `Codestral-22B`, `Nemotron`).
* **Partner Tier Access Control**: Partner models like `mistralai/codestral-22b-instruct-v0.1` require tier-restricted API keys; standard keys return `404 Function Not Found for Account`.

---

### 6.2 Comparative Empirical Experiments Summary

| Experiment Trial | Configured Model Mix (Classifier / Reasoning / Verifier / Fix) | Total Latency (s) | ReAct Search Quality | Verifier Stability | Primary Failure Mode / Key Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trial 1: Monolithic High-Capacity Baseline** | `z-ai/glm-5.2` (All Nodes) | **353.9s** (~5.9m) | **Excellent** (6 Turns) | **100% Grounded** | Successful execution & high precision, but bottlenecked by 10s intent classification overhead. |
| **Trial 2: Heavy 70B Parameter Mix** | `llama-3.1-8b` / `llama-3.3-70b` / `llama-3.3-70b` | **1,324.0s** (~22.0m) | **Good** (6 Turns) | **Grounded** | **Catastrophic Queue Latency**: Shared 70B endpoint suffered 198.3s server wait time per turn. |
| **Trial 3: Light 8B Uniform Mix** | `llama-3.1-8b` (All Nodes) | **116.6s** (~1.9m) | **Failed** (15 Turns) | **Repetitive Loop** | **ReAct Loop Failure**: 8B model repeated identical searches 10 times, bloating prompt context to 15,000 tokens & causing verifier hallucination loops. |
| **Trial 4: Optimized Hybrid Architecture** | `llama-3.1-8b` / `glm-5.2` / `llama-3.1-8b` / `codestral` | **~180s** (~3.0m) | **Optimal** (4-6 Turns) | **100% Grounded** | **Optimal Trade-Off**: 0.3s intent routing + intelligent multi-turn search with zero loops. |

---

### 6.3 Detailed Analysis of Failed & Successful Experiments

#### A. Failure Analysis: The 8B Multi-Turn Reasoning Loop (Trial 3)
* **Observed Symptom**: When evaluating the query `"what LLM and it's framework are we using in it?"`, `meta/llama-3.1-8b-instruct` in `reasoning_node` issued the exact search query `search("NVIDIA NIM model" OR "Cloudflare AI model")` **10 times sequentially** across Iterations 5 to 15.
* **Root Cause Analysis**: Small parameter models (8B) lack high-level planning meta-cognition. When a codebase does not contain literal verbatim string matches for a search term, an 8B model fails to stop searching and instead re-executes redundant searches until reaching the max loop limit (15 iterations).
* **Cascade Effect**: Accumulating 15 turns of raw search observations inflated prompt memory to **15,000 tokens**. Passing this massive context to `verifier_node` caused autoregressive token repetition, outputting over 100 identical bullet points of unsupported claims.

#### B. Failure Analysis: 70B Server Queue Bottleneck (Trial 2)
* **Observed Symptom**: Total execution time exploded to **1,323.998 seconds (22 minutes)**.
* **Root Cause Analysis**: On shared public API endpoints, 70B models (`meta/llama-3.3-70b-instruct`) experience high time-to-first-token (TTFT) and queue delays averaging **198.3 seconds per turn**. Over 6 iterations, queue wait time dominated 90% of total runtime.

#### C. Success Analysis: 8B Intent Classifier & Category 2 Bug Fix Engine (Trial 4)
* **Instant Routing**: Using `meta/llama-3.1-8b-instruct` for `intent_classifier_node` reduced classification latency from **10.28 seconds to 0.31 seconds (33x acceleration)** with 100% accuracy.
* **Bug Fix & Sandbox Verification**: In Category 2 bug fix mode, `fix_proposal_node` generated a complete patch diff in **25.8 seconds**, and `execution_verifier_node` passed AST syntax validation (`ast.parse`) and launched a dynamic pytest unit test script inside a temp sandbox in **11.9 seconds** (total Category 2 latency: **75.7 seconds**).

---

### 6.4 Core Research Conclusions

1. **Do NOT use 8B models for multi-turn ReAct reasoning loops** in complex repositories. 8B models are prone to infinite search loops and prompt context bloat.
2. **Do use 8B models for single-turn structured tasks** (`intent_classifier`, quick extraction, AST syntax validation).
3. **Use high-capacity models (`z-ai/glm-5.2` / 100B+) for `reasoning_node`**. They complete ReAct navigation in 4–5 turns with zero search loops, optimizing both accuracy and real-world execution speed.

---

## 7. Technology Stack

- **Agent Orchestration**: `langgraph` (v1.1.9)
- **AST Parser**: `tree-sitter` (v0.26.0) & `tree-sitter-python` (v0.25.0)
- **Vector Database**: `chromadb` (v1.5.9)
- **Sparse Retrieval**: `rank_bm25` (v0.2.2)
- **Sandbox Subprocess Engine**: `subprocess` + `pytest` (v8.x) + `tempfile`
- **LLM Provider**: **NVIDIA NIM API** (`integrate.api.nvidia.com/v1`) via `openai` Python SDK
- **Embedding Model**: `nvidia/llama-nemotron-embed-1b-v2` via NVIDIA NIM API

---

## 8. Setup & Execution

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
