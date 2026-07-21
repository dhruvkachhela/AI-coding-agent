# Codebase RAG Agent with Grounding Verifier

A production-grade, ReAct-style codebase QA agent built from scratch using **LangGraph**, **Tree-sitter**, **ChromaDB**, **BM25**, and a secondary **Grounding Critic / Verifier LLM**. The agent dynamically navigates repositories using hybrid retrieval (dense embedding + field-weighted BM25 with Reciprocal Rank Fusion) and enforces low-hallucination guarantees via an automated verification feedback loop.

The system was evaluated and tested end-to-end against **VibeCheck Scan / VibeSec Pipeline** (a 9-layer AI-driven static application security testing scanner containing cross-file imports, database schemas, and AI-triage engines).

---

## 1. Motivation & Technical Focus

While traditional RAG pipelines rely on single-shot document retrieval and direct LLM generation, codebase QA presents two unique challenges:
1. **Structural Blind Spots**: High-level architectural flows (e.g. entry points, call-graph chains) cannot be captured by single-turn semantic search alone.
2. **LLM Hallucinations**: Standard ReAct agents often synthesize plausibly sounding but unverified claims (e.g., hallucinating API endpoint paths, missing wrapper layers, or misrepresenting class inheritance).

This project resolves both challenges by implementing a **multi-turn ReAct state machine** paired with a **secondary Grounding Critic LLM** that audits final answers against raw retrieved code observations before returning them to the user.

---

## 2. System Architecture

```
                    +------------------------------------+
                    |        Input User Question         |
                    +-------------------+----------------+
                                        |
                                        v
                    +-------------------+----------------+
     +------------->|          Reasoning Node            |<----------------+
     |              |          (GLM-5.2 LLM)             |                 |
     |              +-------------------+----------------+                 |
     |                                  |                                  |
     |                    [Conditional Edge Decision]                      |
     |                                  |                                  |
     |                 Is there a Final Answer?                            |
     |                   /                     \                           |
     |                 Yes                      No (Search Action)         |
     |                 /                         \                         |
     |                v                           v                        |
     |        +-------+-------+           +-------+--------+               |
     |        | Verifier Node |           |   Tool Node    |               |
     |        | (Critic LLM)  |           | (Hybrid Search)|               |
     |        +-------+-------+           +-------+--------+               |
     |                |                           |                        |
     |         [Grounding Check]                  v                        |
     |          /           \             +-------+--------+               |
     |     Supported    Unsupported       | Dense  + BM25  |               |
     |       /                 \          |   RRF Fusion   |               |
     |      v                   v         +-------+--------+               |
     |   +-----+        (Self-Correction:         |                        |
     |   | END |         Reset Final Answer       |                        |
     |   +-----+         & Inject Feedback)-------+                        |
     |                                            |                        |
     +--------------------------------------------+                        |
                         Appends Observation to State                      |
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
  RRF_Score(d)=(1/(k+Rankdense​(d))) ​+ (1/(k+RankBM25​(d)1))​
---

## 2.4 Grounding Critic & Verifier LLM (Anti-Hallucination Loop)

To eliminate hallucinations, we added an automated verification step to the LangGraph state machine:

### A. State Schema (`AgentState`)
```python
class AgentState(TypedDict):
    question: str
    history: List[Tuple[str, str, str]]  # (thought, action, observation)
    current_thought: str
    action_query: Optional[str]
    final_answer: Optional[str]
    iterations: int
    verification_attempts: int
    verifier_feedback: Optional[str]
    is_grounded: bool
```

### B. The Verifier Node (`verifier_node`)
When `reasoning_node` produces a `Final Answer`, the graph routes to `verifier_node`. 
- **Unbiased Context**: The verifier receives **ONLY** the proposed `Final Answer` and the consolidated `Retrieved Observations` across all search turns (intermediate thoughts and queries are excluded).
- **Prompt Specification**: The critic (`z-ai/glm-5.2` at temperature 0.0) is instructed:
  > *"Check if EVERY claim made in the proposed Final Answer is explicitly supported by the provided retrieved code observations. Respond with `VERDICT: SUPPORTED` or `VERDICT: UNSUPPORTED` followed by a list of unsupported claims."*

### C. Self-Correction Routing (`route_verification`)
- **If Grounded**: The verifier returns `is_grounded = True`, routing the graph to `END`.
- **If Unsupported & Attempts < 2**:
  - The verifier sets `final_answer = None` and populates `verifier_feedback`.
  - The graph routes back to `reasoning_node`.
  - The system prompt injects a `CRITICAL ATTENTION - PREVIOUS ANSWER REJECTED BY VERIFIER` section containing the critic's exact feedback, prompting the model to search for missing details or correct unverified claims.
- **If Unsupported & Attempts >= 2**:
  - The graph appends `[WARNING: Partially Grounded]` with the verifier feedback to the final answer and routes to `END` to prevent infinite loops.

### D. Robust Verdict Parsing
To handle minor LLM formatting/spelling variations (such as `VERDICT: SUPPORTD`), the router uses:
```python
upper_text = verifier_text.upper()
is_grounded = "VERDICT: SUPPORT" in upper_text and "VERDICT: UNSUPPORT" not in upper_text
```

---

## 3. Empirical Evaluation & Trace Verification

### Case Study 1: API Endpoint Verification
* **User Question**: `"what LLM and it's framework are we using in it?"`
* **Iteration 6 (Initial Final Answer)**: The agent generated a detailed answer, but hallucinated that the Cloudflare fallback endpoint was `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions`.
* **Critic Evaluation (Attempt 1)**:
  ```text
  VERDICT: UNSUPPORTED
  Unsupported Claims:
  - The claim that the Cloudflare API endpoint is .../ai/v1/chat/completions is NOT supported by the retrieved code. The actual endpoint in the code is .../ai/run/@cf/meta/llama-3.1-8b-instruct-fast.
  ```
* **Self-Correction (Iteration 7)**: The agent received the critic feedback, corrected the endpoint URL to `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/llama-3.1-8b-instruct-fast`, and resubmitted.
* **Critic Evaluation (Attempt 2)**: `VERDICT: SUPPORTED` $\rightarrow$ Answer delivered to user with 100% precision.

---

### Case Study 2: Taint Analysis Oracle Query
* **User Question**: `"joern is used or not? if yes then how it's working here?"`
* **Iteration 2 (Initial Final Answer)**: The agent generated an answer claiming Joern builds AST/CFG/PDG structures and assumes `/tmp` hardcoded file paths.
* **Critic Evaluation (Attempt 1)**:
  ```text
  VERDICT: UNSUPPORTED
  Unsupported Claims:
  - The code uses tempfile.gettempdir(), not hardcoded /tmp.
  - No snippet explicitly describes CPG as a combination of AST+CFG+PDG.
  ```
* **Self-Correction (Iteration 3)**: The agent revised the explanation to strictly reference `build_global_cpg` in `layer0_indexer.py` and `_tier3_joern` in `layer7_validator.py`.
* **Critic Evaluation (Attempt 2)**: Verified and accepted.

---

## 4. Technology Stack

- **Agent Orchestration**: `langgraph` (v1.1.9)
- **Parser**: `tree-sitter` (v0.26.0) & `tree-sitter-python` (v0.25.0)
- **Vector Database**: `chromadb` (v1.5.9)
- **Sparse Retrieval**: `rank_bm25` (v0.2.2)
- **LLM Engine**: `z-ai/glm-5.2` via **NVIDIA NIM API** (OpenAI SDK client)
- **Embedding Model**: `nvidia/llama-nemotron-embed-1b-v2` via **NVIDIA NIM API**

---

## 5. Repository Structure

```
├── ai-coding-agent.ipynb   # Main Jupyter Notebook containing complete pipeline & cells 0-10
├── repo_indexer.py         # Standalone repository parser, AST chunker, & vector builder
├── README.md               # Architecture documentation & technical specifications
└── codebase_rag_report.html # Evaluation report & performance analysis
```

---

## 6. Setup & Execution

### Prerequisites
Set your NVIDIA NIM API key:
```bash
export NVIDIA_API_KEY="nvapi-..."
```

### Install Dependencies
```bash
pip install tree-sitter tree-sitter-python chromadb rank_bm25 langgraph openai tqdm
```

### Running in Kaggle / Jupyter
Open `ai-coding-agent.ipynb` and run the cells sequentially:
- **Cell 2–5**: AST Chunker, Vector Embedder, and Hybrid Search Engine functions.
- **Cell 9**: `build_agent_graph()` definition with Grounding Verifier state machine.
- **Cell 10**: Execution runner testing sample questions.
