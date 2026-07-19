# Codebase RAG Agent

A ReAct-style coding agent built from scratch using **LangGraph**, **Tree-sitter**, **ChromaDB**, and **BM25** that performs codebase Q&A. The agent is generic and can be pointed at any repository; it uses a hybrid retrieval tool (dense embedding + field-weighted BM25 search) to answer complex codebase queries.

The system was evaluated and tested end-to-end against the private repository **VibeCheck Scan / VibeSec Pipeline** (a 9-layer AI-driven static application security testing scanner containing cross-file imports, database schemas, and AI-triage engines).

---

## 1. Why This Project

Prior projects (`qlora-nl2sql`, `nl2sql-lora`, `flatland_gpt`) focused heavily on transformer internals, QLoRA fine-tuning, and model quantization. This project closes a key gap by focusing on **agentic reasoning and retrieval systems**—specifically, building a multi-turn ReAct loop that can dynamically discover and synthesize context across a codebase, rather than relying on a single flat retrieval stage.

---

## 2. Architecture

```
                    +------------------------------------+
                    |        Input User Question         |
                    +-------------------+----------------+
                                        |
                                        v
                    +-------------------+----------------+
     +------------->|          Reasoning Node            |<------------+
     |              |          (GLM-5.2 LLM)             |             |
     |              +-------------------+----------------+             |
     |                                  |                              |
     |                    [Conditional Edge Decision]                  |
     |                                  |                              |
     |                 Is there a Final Answer?                        |
     |                   /                     \                       |
     |                 Yes                      No (Search Action)     |
     |                 /                         \                     |
     |                v                           v                    |
     |            +---+----+              +-------+--------+           |
     |            |  END   |              |   Tool Node    |           |
     |            +--------+              | (Hybrid Search)|           |
     |                                    +-------+--------+           |
     |                                            |                    |
     |                                            v                    |
     |                                    +-------+--------+           |
     |                                    | Dense  + BM25  |           |
     |                                    |   RRF Fusion   |           |
     |                                    +-------+--------+           |
     |                                            |                    |
     +--------------------------------------------+                    |
                         Appends Observation to State                  |
```

### 2.1 AST-Based Structural Chunking (Method-Level)
Instead of dividing code into arbitrary character windows (which splits function bodies and cuts semantic context), the repository chunker uses **Tree-sitter** to parse Python code into an Abstract Syntax Tree (AST). 
* **The Granularity Decision**: Initially, classes were parsed as single chunks. However, this produced massive outliers: the main `CodeIndexer` class spanned 1,900+ lines (22,956 tokens), creating huge embedding vectors that diluted the details of individual methods.
* **The Refactoring**: The chunker was redesigned to recursively descend into class structures, extracting **individual methods** as separate chunks. To preserve class relationships, each method chunk was prefixed with class-level metadata (e.g., `File: ... \nClass: ... \nMethod: ...`). 
* **The Impact**: This change reduced the maximum chunk size from **22,956 tokens to 7,005 tokens** (corresponding to the monolithic `run_full_scan` function) and reduced the average chunk size from **707.18 to 422.77 tokens**.

### 2.2 Embedding Model Selection
We evaluated three embedding options to find a model capable of supporting the codebase's long-tail distribution without chunk truncation:
1. `sentence-transformers/all-MiniLM-L6-v2`: Context limit of 256 tokens. Truncated over 70% of our codebase chunks.
2. `nvidia/nv-embedcode-v1`: Code-specialized, but limited to a 512-token context. This left a 5% long-tail gap of truncated chunks.
3. `nvidia/llama-nemotron-embed-1b-v2`: General-purpose embedding model trained up to an 8,005-token context window.

**Decision**: We chose `llama-nemotron-embed-1b-v2` because ensuring that 100% of the codebase was fully represented without truncation was more critical for accuracy than using a code-specialized model with context limits. We specify `input_type="passage"` during indexing and `input_type="query"` during retrieval.

### 2.3 Hybrid Search & RRF Fusion
To combine semantic matching with exact keyword lookups, we implemented a hybrid search pipeline:
* **Dense Retrieval**: A persistent local **ChromaDB** instance utilizing Cosine Similarity.
* **Sparse (Keyword) Retrieval**: A field-weighted **BM25** index (`rank_bm25`). Plain single-field BM25 searches penalized long, complex functions due to length normalization. We solved this by splitting BM25 into two separate indexes:
  * `metadata_index` (File path, Class name, Method name), weighted at **3.0**.
  * `body_index` (raw code body), weighted at **1.0**.
* **Fusion**: Results from both streams (top 20 candidates each) are combined using **Reciprocal Rank Fusion (RRF)**:
  $$RRF\_Score = \frac{1}{60 + Rank_{dense}} + \frac{1}{60 + Rank_{BM25}}$$

### 2.4 ReAct Agent Loop (LangGraph)
We wired the search tool into a ReAct loop using LangGraph:
* **State**: A TypedDict tracking the original question, a list of past `(thought, action, observation)` turns, the active query, the current thought, and the final answer.
* **Iteration Cap**: Hard-capped at **5 cycles**. If exceeded, the agent executes a final summary call to synthesize all collected observations, returning a partial answer marked with `"Incomplete — max iterations reached"`.
* **Reasoning Model**: **GLM-5.2** via the NVIDIA NIM API. In informal testing against *Gemini 3.5 Flash*, GLM-5.2 demonstrated significantly lower latency (completing 5-step agent loops in under 10 seconds compared to Llama/Gemini queues) and showed better instruction-following adherence when formatting `Thought:` and `Action:` blocks.

---

## 3. Retrieval Evaluation & Documented Failures

We tested the pure hybrid search pipeline (without the agent loop) on various developer queries. While it achieved strong results on keyword-aligned queries (e.g., retrieving exact regex schemas for "hardcoded secrets" or database lookups for "false positives"), we documented two clear failures:

### Failure 1: The Orchestrator Query
* **Query**: `"find the orchestrator function"`
* **Result**: Pure hybrid search failed to identify `run_full_scan` in `orchestrator.py` as the entry point, topping out at a low similarity score of `0.27 - 0.28`. Instead, irrelevant helper functions within the same file ranked higher.
* **Diagnosis**: The concept of "orchestration" was represented by the file name (`orchestrator.py`) and its structural position in the call graph, not by keyword content inside the function body itself. This is a call-graph fact, not a text similarity property. Changing embedding models or tweaking BM25 field weights could not fix this search gap.

### Failure 2: The LLM Call Site Query
* **Query**: `"which functions call the LLM?"`
* **Result**: Retrieval returned code from `layer1_llm_vulns.py` (which detects LLM vulnerabilities using AST rules) instead of the actual LLM integration site inside `layer9_report.py` (`_enrich_findings_batch`).
* **Diagnosis**: The vulnerability checker contained dense keywords like "llm", "completion", and "response" inside its rule tables, causing it to rank higher than the actual wrapper that calls the OpenAI client.

---

## 4. Agent Success Beyond Retrieval Limits

The ReAct agent successfully resolved the orchestrator failure where single-shot retrieval failed. 

Given the query: `"What is the overall flow from scanning a file to producing a report, and which file orchestrates it?"`, the agent executed the following multi-step reasoning trace:

1. **Step 1 (Reasoning)**: The agent reasoned that it needed to locate the main orchestration process of the pipeline. It called the tool with query: `file scanning report generation orchestrator`.
2. **Step 2 (Tool Observation)**: The tool returned code chunks from `orchestrator.py` showing helper methods like `merge_proximity_findings` and `_merge_subgroup`. 
3. **Step 3 (Reasoning)**: The agent read the code, recognized that these were helper routines inside the orchestration module, and reasoned that it needed to inspect how these layers were coordinated and how reports were generated. It issued a second query: `vibesec pipeline orchestrator layer9 report generation`.
4. **Step 5 (Synthesis & Final Answer)**: The agent successfully mapped the entire sequence:
   * It identified that `orchestrator.py` coordinates the execution flow (from Layer 0 indexing up to Layer 7 validation).
   * It identified that `layer9_report.py` takes the final findings and generates the formatted results.
   * It named `run_full_scan` as the entry function.

**Significance**: This end-to-end flow demonstrates the core value of the agent layer. By executing sequential searches and reasoning about the intermediate code structures, the agent bypassed the limits of single-shot vector searches to resolve structural/call-graph properties.

---

## 5. Known Limitations

* **Call-Graph Blind Spots**: While the agent can resolve call-graph questions via multi-step exploration, the retrieval tool itself remains blind to structural imports. This has only been tested against one specific flow query; we have not validated how well the agent handles complex, nested call-graphs.
* **Informal Benchmarks**: The selection of GLM-5.2 and the embedding model comparison were based on qualitative observations of reasoning traces on our three test queries, rather than a rigorous quantitative benchmark (like MTEB or SWE-bench).
* **Scope Limits**: The agent does not currently propose code fixes or verify changes against a test suite; it is restricted to Q&A.
* **Language Support**: The AST chunker is configured specifically for Python. Generalization to multi-language codebases (e.g., JS/TS, Go) has not been implemented or verified.

---

## 6. What This Project Demonstrates

1. **Granular AST Chunking**: Isolating methods while preserving class relationships using Tree-sitter.
2. **Field-Weighted Hybrid Search**: Combining ChromaDB cosine vectors with separate BM25 metadata and body indexes under Reciprocal Rank Fusion.
3. **State Machine Agent Loops**: A LangGraph state machine driving a ReAct loop with a hard iteration cap.
4. **Agentic Reasoning Value**: A documented case showing how a multi-turn agent successfully resolves structural codebase queries that escape single-turn retrieval.
