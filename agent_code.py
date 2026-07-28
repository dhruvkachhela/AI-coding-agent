!pip install --upgrade opentelemetry-api opentelemetry-sdk chromadb tree-sitter tree-sitter-python openai tqdm rank_bm25 langgraph pytest pytest-mock

import os
import sys
import uuid
import ast
import re
import tempfile
import subprocess
import shutil
from pathlib import Path
import chromadb
import tree_sitter
import tree_sitter_python
from openai import OpenAI
from rank_bm25 import BM25Okapi
from typing import TypedDict, List, Tuple, Optional, Dict, Any
from langgraph.graph import StateGraph, END


def chunk_repo(repo_path):
    """
    Recursively scans directory specified by repo_path for Python (.py) files,
    parses each file using Tree-sitter, and extracts top-level function definitions,
    class methods, and top-level expression statements/assignments (module-level constants).
    """
    chunks = []
    try:
        language = tree_sitter.Language(tree_sitter_python.language())
        parser = tree_sitter.Parser(language)
    except Exception as error:
        print("Error initializing Tree-sitter parser:", error)
        return chunks

    for root_dir, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root_dir, file)
                rel_path = os.path.relpath(full_path, repo_path).replace('\\', '/')
                
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as python_file:
                        content = python_file.read()
                except Exception as error:
                    print("Warning: Failed to read file", rel_path, ":", error)
                    continue

                try:
                    content_bytes = content.encode('utf-8')
                    tree = parser.parse(content_bytes)
                    
                    if tree.root_node.has_error:
                        print("Warning: Syntax errors in file", rel_path, ". Skipping.")
                        continue
                    
                    for child in tree.root_node.children:
                        if child.type == 'function_definition':
                            name_node = child.child_by_field_name('name')
                            name = name_node.text.decode('utf-8', errors='replace') if name_node else 'unknown'
                            code_text = content_bytes[child.start_byte:child.end_byte].decode('utf-8', errors='replace')
                            
                            chunks.append({
                                "code": code_text,
                                "file_path": rel_path,
                                "class_name": "",
                                "name": name,
                                "type": "function",
                                "start_line": child.start_point[0] + 1,
                                "end_line": child.end_point[0] + 1
                            })
                        
                        elif child.type == 'class_definition':
                            class_name_node = child.child_by_field_name('name')
                            class_name = class_name_node.text.decode('utf-8', errors='replace') if class_name_node else 'unknown'
                            body_node = child.child_by_field_name('body')
                            
                            if body_node:
                                for body_child in body_node.children:
                                    if body_child.type == 'function_definition':
                                        method_name_node = body_child.child_by_field_name('name')
                                        method_name = method_name_node.text.decode('utf-8', errors='replace') if method_name_node else 'unknown'
                                        method_code = content_bytes[body_child.start_byte:body_child.end_byte].decode('utf-8', errors='replace')
                                        
                                        chunks.append({
                                            "code": f"# File: {rel_path}\n# Class: {class_name}\n{method_code}",
                                            "file_path": rel_path,
                                            "class_name": class_name,
                                            "name": method_name,
                                            "type": "method",
                                            "start_line": body_child.start_point[0] + 1,
                                            "end_line": body_child.end_point[0] + 1
                                        })
                        
                        elif child.type == 'expression_statement':
                            for sub in child.children:
                                if sub.type == 'assignment':
                                    left_node = sub.child_by_field_name('left')
                                    const_name = left_node.text.decode('utf-8', errors='replace') if left_node else 'unknown'
                                    code_text = content_bytes[child.start_byte:child.end_byte].decode('utf-8', errors='replace')
                                    
                                    chunks.append({
                                        "code": code_text,
                                        "file_path": rel_path,
                                        "class_name": "",
                                        "name": const_name,
                                        "type": "constant",
                                        "start_line": child.start_point[0] + 1,
                                        "end_line": child.end_point[0] + 1
                                    })
                        
                        elif child.type == 'assignment':
                            left_node = child.child_by_field_name('left')
                            const_name = left_node.text.decode('utf-8', errors='replace') if left_node else 'unknown'
                            code_text = content_bytes[child.start_byte:child.end_byte].decode('utf-8', errors='replace')
                            
                            chunks.append({
                                "code": code_text,
                                "file_path": rel_path,
                                "class_name": "",
                                "name": const_name,
                                "type": "constant",
                                "start_line": child.start_point[0] + 1,
                                "end_line": child.end_point[0] + 1
                            })
                except Exception as error:
                    print("Warning: Tree-sitter parse failed for", rel_path, ":", error)
                    
    return chunks


def embed_and_store(chunks, collection_name, persist_dir, nvidia_api_key):
    """
    Converts code chunks into mathematical embeddings using NVIDIA NIM API (llama-3.2-nv-embedqa-1b-v2)
    and stores those embeddings along with their metadata inside a local, persistent Chroma vector database.
    """
    if not chunks:
        print("No chunks to embed and store.")
        return

    # Conceptually, we connect to the NVIDIA NIM API to generate embeddings in the cloud.
    # This prevents downloading heavy model weights (like PyTorch/Transformers) to your laptop.
    print("Initializing NVIDIA NIM API connection...")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_api_key
    )

    # Conceptually, chromadb is a database that stores these high-dimensional vectors and builds a search index
    # (using HNSW - Hierarchical Navigable Small World).
    print("Connecting to persistent Chroma database at:", persist_dir)
    client_db = chromadb.PersistentClient(path=persist_dir)

    # Get or create collection with cosine similarity configuration
    collection = client_db.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )       

    total_chunks = len(chunks)
    batch_size = 50

    from tqdm import tqdm

    print("Starting embedding and storage of", total_chunks, "chunks using NVIDIA NIM...")
    for i in tqdm(range(0, total_chunks, batch_size), desc="Embedding and storing chunks"):
        batch = chunks[i:i + batch_size]
        
        batch_codes = []
        batch_texts_to_embed = []
        for chunk in batch:
            batch_codes.append(chunk['code'])
            # Create a structured prefix text that includes the class name for method chunks
            if chunk['type'] == 'method':
                text_to_embed = "File: " + chunk['file_path'] + "\nClass: " + chunk['class_name'] + "\nMethod: " + chunk['name'] + "\nType: method\n\n" + chunk['code']
            elif chunk['type'] == 'class_definition':
                text_to_embed = "File: " + chunk['file_path'] + "\nClass: " + chunk['class_name'] + "\nType: class_definition\n\n" + chunk['code']
            else:
                text_to_embed = "File: " + chunk['file_path'] + "\nName: " + chunk['name'] + "\nType: function\n\n" + chunk['code']
            batch_texts_to_embed.append(text_to_embed)

        # Call NVIDIA NIM API to generate embeddings for the batch
        response = client.embeddings.create(
            input=batch_texts_to_embed,
            model="nvidia/llama-nemotron-embed-1b-v2",
            extra_body={"input_type": "passage", "truncate": "NONE"}
        )
        
        # Extract embeddings list from response
        batch_embeddings = []
        for data in response.data:
            batch_embeddings.append(data.embedding)

        # Generate unique IDs and extract metadata
        batch_ids = []
        for _ in batch:
            batch_ids.append(uuid.uuid4().hex)
            
        batch_metadatas = []
        for chunk in batch:
            metadata_dict = {
                "file_path": chunk["file_path"],
                "class_name": chunk["class_name"],
                "name": chunk["name"],
                "type": chunk["type"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"]
            }
            batch_metadatas.append(metadata_dict)

        # Add batch to Chroma DB collection
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_codes
        )


def query_repo(query, collection_name, persist_dir, nvidia_api_key, top_k=5):
    """
    Converts a natural language query into a vector embedding using NVIDIA NIM and searches the
    Chroma vector database to find the top_k most semantically similar code chunks.
    """
    # Connect to NVIDIA NIM API
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_api_key
    )
    
    # Call NVIDIA NIM API to convert query to vector representation.
    response = client.embeddings.create(
        input=query,
        model="nvidia/llama-nemotron-embed-1b-v2",
        extra_body={"input_type": "query"}
    )
    query_embedding = response.data[0].embedding

    # Initialize Chroma client
    client_db = chromadb.PersistentClient(path=persist_dir)

    # Retrieve collection
    try:
        collection = client_db.get_collection(name=collection_name)
    except Exception as error:
        print("Error: Collection '", collection_name, "' not found:", error)
        return []

    # Query Chroma using query embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    output = []
    if results is not None:
        if 'documents' in results and results['documents'] is not None:
            if len(results['documents']) > 0:
                docs = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]

                # Map results to our output format
                for idx in range(len(docs)):
                    document_text = docs[idx]
                    metadata = metadatas[idx]
                    distance = distances[idx]
                    
                    # Calculate cosine similarity (1.0 - cosine distance)
                    similarity = 1.0 - distance
                    
                    match_dict = {
                        "code": document_text,
                        "file_path": metadata["file_path"],
                        "class_name": metadata.get("class_name", ""),
                        "name": metadata["name"],
                        "type": metadata.get("type", "function"),
                        "start_line": int(metadata.get("start_line", 0)),
                        "end_line": int(metadata.get("end_line", 0)),
                        "similarity_score": similarity
                    }
                    output.append(match_dict)

    return output


def tokenize_text(text):
    """
    Splits a raw text string into a list of lowercase alphanumeric word tokens.
    """
    words = []
    current_word = []
    for char in text:
        if char.isalnum():
            current_word.append(char.lower())
        else:
            if current_word:
                words.append("".join(current_word))
                current_word = []
    if current_word:
        words.append("".join(current_word))
    return words

def build_bm25_index(chunks):
    """
    Constructs two separate BM25Okapi search indexes over the extracted codebase chunks:
    one for metadata (file path, class, method/function name) and one for the code body.
    """
    metadata_corpus = []
    body_corpus = []
    for chunk in chunks:
        # Build metadata string based on chunk type
        if chunk['type'] == 'method':
            metadata_str = "File: " + chunk['file_path'] + "\nClass: " + chunk['class_name'] + "\nMethod: " + chunk['name']
        elif chunk['type'] == 'class_definition':
            metadata_str = "File: " + chunk['file_path'] + "\nClass: " + chunk['class_name']
        else:
            metadata_str = "File: " + chunk['file_path'] + "\nName: " + chunk['name']
            
        metadata_corpus.append(tokenize_text(metadata_str))
        body_corpus.append(tokenize_text(chunk['code']))
        
    metadata_index = BM25Okapi(metadata_corpus)
    body_index = BM25Okapi(body_corpus)
    return metadata_index, body_index, chunks

def bm25_search(query, metadata_index, body_index, chunks, top_k=10, metadata_weight=3.0, body_weight=1.0):
    """
    Ranks chunks using a field-weighted combination of metadata and code body BM25 scores.
    """
    tokenized_query = tokenize_text(query)
    
    # Get scores from both indexes
    metadata_scores = metadata_index.get_scores(tokenized_query)
    body_scores = body_index.get_scores(tokenized_query)
    
    scored_chunks = []
    for idx in range(len(chunks)):
        m_score = float(metadata_scores[idx])
        b_score = float(body_scores[idx])
        combined_score = (metadata_weight * m_score) + (body_weight * b_score)
        
        scored_chunks.append({
            "chunk": chunks[idx],
            "score": combined_score
        })
        
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    results = []
    for item in scored_chunks[:top_k]:
        chunk = item["chunk"]
        results.append({
            "code": chunk["code"],
            "file_path": chunk["file_path"],
            "class_name": chunk.get("class_name", ""),
            "name": chunk["name"],
            "type": chunk["type"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "bm25_score": item["score"]
        })
    return results

def hybrid_search(query, collection_name, persist_dir, metadata_index, body_index, chunks, nvidia_api_key, top_k=5, rrf_k=60, metadata_weight=3.0, body_weight=1.0):
    """
    Fuses dense embedding similarity and field-weighted sparse BM25 keyword matching scores
    using Reciprocal Rank Fusion (RRF).
    """
    # 1. Fetch top 20 candidates from dense search
    dense_results = query_repo(query, collection_name, persist_dir, nvidia_api_key, top_k=20)
    
    # 2. Fetch top 20 candidates from BM25 search
    bm25_results = bm25_search(
        query=query,
        metadata_index=metadata_index,
        body_index=body_index,
        chunks=chunks,
        top_k=20,
        metadata_weight=metadata_weight,
        body_weight=body_weight
    )
    
    # Track rank and scores for both lists
    dense_rank = {}
    dense_scores = {}
    for idx in range(len(dense_results)):
        r = dense_results[idx]
        key = r["file_path"] + "|" + r["name"] + "|" + str(r["start_line"])
        dense_rank[key] = idx + 1
        dense_scores[key] = r["similarity_score"]
        
    bm25_rank = {}
    bm25_scores = {}
    for idx in range(len(bm25_results)):
        r = bm25_results[idx]
        key = r["file_path"] + "|" + r["name"] + "|" + str(r["start_line"])
        bm25_rank[key] = idx + 1
        bm25_scores[key] = r["bm25_score"]
        
    # Combine keys from both lists
    all_keys = set(dense_rank.keys()).union(set(bm25_rank.keys()))
    
    fused_results = []
    for key in all_keys:
        d_rank = dense_rank.get(key)
        rrf_dense = 1.0 / (rrf_k + d_rank) if d_rank is not None else 0.0
            
        b_rank = bm25_rank.get(key)
        rrf_bm25 = 1.0 / (rrf_k + b_rank) if b_rank is not None else 0.0
            
        rrf_score = rrf_dense + rrf_bm25
        
        # Retrieve the original chunk metadata from either of the result lists
        matched_result = None
        for r in dense_results:
            r_key = r["file_path"] + "|" + r["name"] + "|" + str(r["start_line"])
            if r_key == key:
                matched_result = r
                break
        if matched_result is None:
            for r in bm25_results:
                r_key = r["file_path"] + "|" + r["name"] + "|" + str(r["start_line"])
                if r_key == key:
                    matched_result = r
                    break
                    
        if matched_result is not None:
            fused_results.append({
                "code": matched_result["code"],
                "file_path": matched_result["file_path"],
                "class_name": matched_result["class_name"],
                "name": matched_result["name"],
                "type": matched_result.get("type", "function"),
                "start_line": matched_result["start_line"],
                "end_line": matched_result["end_line"],
                "dense_score": dense_scores.get(key, 0.0),
                "bm25_score": bm25_scores.get(key, 0.0),
                "rrf_score": rrf_score
            })
            
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_results[:top_k]

import os

# Your GitHub token and repository details
token = "[REDACTED_GITHUB_TOKEN]"
github_username = "dhruvkachhela"
repo_name = "vibesec"

# Clone the repository directly onto the Kaggle server
print("Cloning private repository...")
exit_code = os.system(f"git clone https://{token}@github.com/dhruvkachhela/vibecheck-scan")

if exit_code == 0:
    print("Success! Repository cloned successfully.")
else:
    print("Error: Failed to clone repository. Please check your username and repository name.")


import os
import shutil
from kaggle_secrets import UserSecretsClient

# Generic Kaggle working folder or local working directory fallback
if os.path.exists("/kaggle/working/vibecheck-scan"):
    repo_path = "/kaggle/working/vibecheck-scan"
elif os.path.exists("./vibecheck-scan"):
    repo_path = "./vibecheck-scan"
else:
    repo_path = "."

collection_name = "repo_code_chunks"
persist_dir = "./chroma_db"

# Fetch NVIDIA API Key (Kaggle secrets or hardcoded fallback)
try:
    user_secrets = UserSecretsClient()
    nvidia_api_key = user_secrets.get_secret("nvidia_api_key")
except Exception:
    nvidia_api_key = "nvapi-YOUR_KEY_HERE"  # Set your hardcoded NVIDIA API Key string here if needed

# Automatically clean up old database folder before rebuilding to prevent duplicates
shutil.rmtree(persist_dir, ignore_errors=True)

# 1. Extract chunks
print("Step 1: Chunking codebase inside:", repo_path)
chunks = chunk_repo(repo_path)
print("Successfully extracted", len(chunks), "code chunks.")

# 2. Embed and store using NVIDIA NIM API
if chunks:
    print("\nStep 2: Embedding chunks and storing in Chroma...")
    embed_and_store(chunks, collection_name, persist_dir, nvidia_api_key)
    
    # 3. Build sparse keyword indexes
    print("\nStep 3: Compiling BM25 keyword indexes...")
    metadata_index, body_index, bm25_chunks = build_bm25_index(chunks)
    print("BM25 indexes built successfully.")
    
    print("\nIndexing completed successfully!")


# Configure same collection details
collection_name = "repo_code_chunks"
persist_dir = "./chroma_db"

# Fetch NVIDIA API Key directly inside this cell
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    nvidia_api_key = user_secrets.get_secret("nvidia_api_key")
except Exception:
    nvidia_api_key = "your-nvapi-key-here"

# List of queries you want to run
queries = [
    "find the orchestrator function",
    "how does the system decide what's a false positive?",
    "which functions call the LLM?"
]

# Loop through each query and print results
for query in queries:
    print("\n" + "=" * 80)
    print(f"HYBRID QUERY: '{query}'")
    print("=" * 80)
    
    # query top_k=2 results, setting metadata_weight=3.0 and body_weight=1.0
    results = hybrid_search(
        query=query,
        collection_name=collection_name,
        persist_dir=persist_dir,
        metadata_index=metadata_index,
        body_index=body_index,
        chunks=bm25_chunks,
        nvidia_api_key=nvidia_api_key,
        top_k=2,
        rrf_k=60,
        metadata_weight=3.0,
        body_weight=1.0
    )
    
    if not results:
        print("No matches retrieved.")
    else:
        for r_idx in range(len(results)):
            result = results[r_idx]
            display_idx = r_idx + 1
            print(f"\nResult {display_idx} (RRF Score: {round(result['rrf_score'], 5)} | Dense Score: {round(result['dense_score'], 4)} | BM25 Score: {round(result['bm25_score'], 4)}):")
            print("File:", result['file_path'])
            if result['class_name']:
                print("Class:", result['class_name'], "| Name:", result['name'])
            else:
                print("Name:", result['name'])
            print("-" * 40)
            
            lines = result['code'].splitlines()
            snippet_lines = []
            for line_idx in range(min(10, len(lines))):
                snippet_lines.append(lines[line_idx])
            snippet = "\n".join(snippet_lines)
            
            if len(lines) > 10:
                snippet += "\n..."
            print(snippet)

"""
How this works:
This module sets up a Dual-Mode Agent pipeline using LangGraph and OpenAI/NVIDIA API clients.
1. Defines ANSI color codes for formatted terminal output during execution.
2. Defines `extract_message_content` to safely extract content from OpenAI responses,
   handling thinking/reasoning models (like step-3.7-flash and inkling) without returning None.
3. Defines helper functions for latency tracking (`record_latency`), code formatting (`clean_code_snippet`),
   and benchmark summary reporting (`print_latency_benchmark_report`).
4. Defines `build_agent_graph`, which builds a state graph consisting of:
   - `intent_classifier_node`: Classifies user input into QA or FIX_PROPOSAL (with 512 token budget).
   - `reasoning_node`: Decides whether to query codebase search tools or finalize answer.
   - `tool_node`: Executes hybrid search (AST + BM25 + Vector DB) to retrieve code chunks.
   - `verifier_node`: Audits QA answers to ensure they are grounded in retrieved context.
   - `fix_proposal_node`: Generates proposed bug fix code diffs for code repair tasks.
   - `execution_verifier_node`: Runs unit tests in an isolated sandbox environment to verify fixes.
5. Employs `route_reasoning` which guarantees `tool_node` runs for all search actions,
   and `verifier_node` runs ONLY at the end when reasoning produces a Final Answer.
6. Returns the compiled LangGraph workflow ready for invocation via `.invoke()`.
"""

import time
import os
import sys
import json
import re
import shutil
import tempfile
import subprocess
import ast
from typing import TypedDict, Optional, List, Tuple, Dict, Any

# Define ANSI color constants for terminal UI formatting
COLOR_RESET = "\033[0m"
COLOR_GREEN_BOLD = "\033[1;92m"
COLOR_GREEN = "\033[92m"
COLOR_CYAN_BOLD = "\033[1;96m"
COLOR_CYAN = "\033[96m"
COLOR_YELLOW_BOLD = "\033[1;93m"
COLOR_MAGENTA_BOLD = "\033[1;95m"
COLOR_BLUE = "\033[94m"
COLOR_BLUE_BOLD = "\033[1;94m"


def extract_message_content(response) -> str:
    """
    Safely extracts text content from an LLM completion response object.
    Handles thinking/reasoning models where content may be None or stored in reasoning fields.

    Parameters:
    response: The completion response returned by OpenAI client.

    Returns:
    str: The extracted text string, guaranteed to be a string (never None).
    """
    if not response or not getattr(response, "choices", None):
        return ""
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if not message:
        return ""
    if getattr(message, "content", None) is not None:
        return message.content or ""
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return message.reasoning_content or ""
    if hasattr(message, "reasoning") and message.reasoning:
        return message.reasoning or ""
    return ""


class AgentState(TypedDict):
    question: str
    intent_category: Optional[str]
    history: List[Tuple[str, str, str]]
    current_thought: str
    action_query: Optional[str]
    final_answer: Optional[str]
    proposed_fix: Optional[Dict[str, Any]]
    sandbox_test_script: Optional[str]
    test_results: Optional[Dict[str, Any]]
    iterations: int
    verification_attempts: int
    verifier_feedback: Optional[str]
    is_grounded: bool
    node_latencies: Dict[str, List[float]]


def clean_code_snippet(code_snippet: str) -> str:
    """
    Strips markdown code block formatting (e.g. ```python ... ```) from a code string.

    Parameters:
    code_snippet (str): The raw string extracted from an LLM response.

    Returns:
    str: Cleaned code string ready for AST parsing or execution.
    """
    if not code_snippet:
        return ""
    cleaned = code_snippet.strip()
    cleaned = re.sub(r"^```(?:python|py)?\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def call_with_retry(client, max_retries=4, base_delay=2.0, **kwargs):
    """Call chat.completions.create with exponential backoff on rate limits/transient errors."""
    if "minimax" in kwargs.get("model", ""):
        kwargs["extra_body"] = {"chat_template_kwargs": {"thinking_mode": "disabled"}}
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            is_last = attempt == max_retries - 1
            if is_last:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"      [LLM RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
            import time
            time.sleep(delay)


def record_latency(state: AgentState, node_name: str, total_duration: float, api_duration: float = 0.0) -> Dict[str, List[Dict[str, float]]]:
    """
    Records timing benchmarks for a specific node execution within the agent state.

    Parameters:
    state (AgentState): The current graph state containing existing latencies.
    node_name (str): The identifier name of the agent node.
    total_duration (float): Total wall-clock time spent in the node (seconds).
    api_duration (float): Time spent waiting for external API calls (seconds).

    Returns:
    Dict[str, List[Dict[str, float]]]: Updated latency dictionary for state propagation.
    """
    latencies = dict(state.get("node_latencies", {}))
    if node_name not in latencies:
        latencies[node_name] = []
    internal_duration = max(0.0, total_duration - api_duration)
    latencies[node_name] = list(latencies[node_name]) + [{
        "total": total_duration,
        "api": api_duration,
        "internal": internal_duration
    }]
    return latencies


def print_latency_benchmark_report(state: AgentState, models_info: Optional[Dict[str, str]] = None):
    """
    Prints a structured terminal report detailing latency breakdown for each agent node.

    Parameters:
    state (AgentState): The completed graph execution state containing node latencies.
    models_info (Optional[Dict[str, str]]): Mapping of node names to model names for reporting.

    Returns:
    None
    """
    latencies = state.get("node_latencies", {})
    total_pipeline_time = sum(sum(item["total"] for item in durations) for durations in latencies.values())
    
    print("\n" + "=" * 105)
    print(f"{COLOR_MAGENTA_BOLD}                 DUAL-MODE AGENT LATENCY & PERFORMANCE BENCHMARK REPORT{COLOR_RESET}")
    print("=" * 105)
    print(f"{'Node Name':<20} | {'Model ID':<28} | {'Calls':<5} | {'Total (s)':<9} | {'API (s)':<9} | {'Internal':<9} | {'Avg (s)':<8} | {'% Total':<7}")
    print("-" * 105)
    
    for node_name, durations in latencies.items():
        if not durations:
            continue
        call_count = len(durations)
        total_time = sum(item["total"] for item in durations)
        api_time = sum(item["api"] for item in durations)
        internal_time = sum(item["internal"] for item in durations)
        avg_time = total_time / call_count if call_count > 0 else 0.0
        pct = (total_time / total_pipeline_time * 100) if total_pipeline_time > 0 else 0.0
        model_name = models_info.get(node_name, "N/A") if models_info else "N/A"
        print(f"{node_name:<20} | {model_name:<28} | {call_count:<5} | {total_time:<9.3f} | {api_time:<9.3f} | {internal_time:<9.3f} | {avg_time:<8.3f} | {pct:<6.1f}%")
        
    print("-" * 105)
    print(f"{COLOR_CYAN_BOLD}Total Agent Execution Latency: {total_pipeline_time:.3f} seconds{COLOR_RESET}")
    print("=" * 105 + "\n")


def build_agent_graph(
    collection_name: str,
    persist_dir: str,
    metadata_index: Any,
    body_index: Any,
    bm25_chunks: List[Dict[str, Any]],
    nvidia_api_key: str,
    model_classifier: str = "mistralai/mistral-medium-3.5-128b",
    model_reasoning: str = "z-ai/glm-5.2",
    model_verifier: str = "thinkingmachines/inkling",
    model_fix_proposal: str = "minimaxai/minimax-m3",
    model_test_generator: str = "minimaxai/minimax-m3"
):
    """
    Constructs and compiles the multi-agent LangGraph application for codebase search and bug fixing.

    Parameters:
    collection_name (str): Name of vector DB collection.
    persist_dir (str): Path to persistent storage directory.
    metadata_index (Any): Pre-built metadata search index.
    body_index (Any): Pre-built body content search index.
    bm25_chunks (List[Dict[str, Any]]): Chunks structured for BM25 retrieval.
    nvidia_api_key (str): API key for NVIDIA API endpoint access.
    model_classifier (str): LLM identifier for intent classification.
    model_reasoning (str): LLM identifier for reasoning node.
    model_verifier (str): LLM identifier for answer verification.
    model_fix_proposal (str): LLM identifier for code fix synthesis.
    model_test_generator (str): LLM identifier for sandbox unit test generation.

    Returns:
    Compiled StateGraph application executable via `.invoke()`.
    """
    configured_models = {
        "intent_classifier": model_classifier,
        "reasoning": model_reasoning,
        "tool": "hybrid_search (AST+BM25+DB)",
        "verifier": model_verifier,
        "fix_proposal": model_fix_proposal,
        "execution_verifier": model_test_generator
    }

    def intent_classifier_node(state: AgentState):
        start_time = time.time()
        question = state["question"]
        
        system_prompt = (
            "You are a specialized Intent Classifier for an enterprise Codebase RAG system.\n"
            "Analyze the user request and classify it into EXACTLY one category:\n\n"
            "1. QA: The user is asking an informational, architectural, or explanatory question about how the codebase operates.\n"
            "2. FIX_PROPOSAL: The user is reporting a bug, asking to fix an issue, patch code, or resolve an exception.\n\n"
            "Respond ONLY with a single word: QA or FIX_PROPOSAL."
        )
        
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        
        api_start = time.time()
        response = call_with_retry(client, 
            model=model_classifier,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Request: {question}"}
            ],
            temperature=0.0,
            max_tokens=512
        )
        api_elapsed = time.time() - api_start
        total_elapsed = time.time() - start_time
        internal_elapsed = max(0.0, total_elapsed - api_elapsed)
        
        raw_text = extract_message_content(response)
        category = raw_text.strip().upper()
        if "FIX" in category:
            intent = "FIX_PROPOSAL"
        else:
            intent = "QA"
            
        updated_latencies = record_latency(state, "intent_classifier", total_elapsed, api_elapsed)
        print(f"\n--- [{COLOR_MAGENTA_BOLD}INTENT CLASSIFIER AGENT{COLOR_RESET}] (Model: {model_classifier} | API: {api_elapsed:.3f}s | Internal: {internal_elapsed:.3f}s | Total: {total_elapsed:.3f}s) ---")
        print(f"Classified User Intent: {intent}")
        
        return {
            "intent_category": intent,
            "node_latencies": updated_latencies
        }

    def reasoning_node(state: AgentState):
        start_time = time.time()
        iterations = state.get("iterations", 0) + 1
        history_list = state.get("history", [])
        verifier_feedback = state.get("verifier_feedback", None)
        
        history_str = ""
        for i, (thought, action, obs) in enumerate(history_list):
            history_str += f"\nTurn {i+1}:\nThought: {thought}\nAction: search('{action}')\nObservation:\n{obs}\n"
            
        system_prompt = """You are a Senior Principal Software Architect navigating a complex codebase.
Your objective is to gather necessary code context using precise search queries.

To search the codebase, output EXACTLY:
Thought: [your reasoning for what specific symbol, file, or class to search]
Action: search("[exact search terms or function names]")

When you have retrieved sufficient code context, output EXACTLY:
Thought: [your conclusion that sufficient context has been gathered]
Final Answer: [your response]

CRITICAL: You must output EXACTLY ONE Thought and ONE Action (or ONE Final Answer),
never both, never more than one of each. NEVER write your own "Observation:" lines —
you do not have search results yet. NEVER simulate multiple search turns in a single
response. NEVER write "Turn 2:", "Turn 3:", etc. If you write more than one Action or
any Observation yourself, your entire response will be discarded as invalid.

CRITICAL RESPONSE GUIDELINES FOR FINAL ANSWER:
1. For QA queries: Output a clear, structured, high-level natural language explanation in plain English. Explain the architecture, components, and frameworks clearly. DO NOT dump raw function source code bodies.
2. For FIX_PROPOSAL queries: Clearly identify the target file path, method/function name, line number, and root cause of the bug.
"""
        
        if verifier_feedback and state.get("verification_attempts", 0) > 0:
            system_prompt += f"\n\nCRITICAL ATTENTION - PREVIOUS REJECTION FEEDBACK FROM VERIFIER:\n{verifier_feedback}\n"
            
        user_prompt = f"Question: {state['question']}\n\nSearch History:\n{history_str if history_str else 'No searches conducted yet.'}\n\nDetermine next step."
        
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        
        api_start = time.time()
        response = call_with_retry(client, 
            model=model_reasoning,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        api_elapsed = time.time() - api_start
        
        text = extract_message_content(response).strip()
        
        if text.count("Observation:") > 0 or text.count("Action:") > 1:
            print(f"[{COLOR_YELLOW_BOLD}HALLUCINATION GUARD{COLOR_RESET}] Model generated a fake multi-turn transcript. Truncating to first real step.")
            first_action_idx = text.find("\nAction:")
            if first_action_idx == -1 and text.startswith("Action:"):
                first_action_idx = 0
            if first_action_idx != -1:
                search_start = first_action_idx if first_action_idx == 0 else first_action_idx + 1
                next_line_end = text.find("\n", search_start)
                text = text[:next_line_end if next_line_end != -1 else len(text)]
        
        current_thought = ""
        action_query = None
        final_answer = None
        
        if "Thought:" in text:
            extracted_thought = text.split("Thought:")[1]
            stop_markers = ["Action:", "Final Answer:", "Observation:", "Turn"]
            for marker in stop_markers:
                if marker in extracted_thought:
                    extracted_thought = extracted_thought.split(marker)[0]
            current_thought = extracted_thought.strip()
        else:
            current_thought = text

        if "Action:" in text and "search(" in text:
            after_search = text.split("search(")[1]
            action_query = after_search.split(")")[0].strip("\"' ")

        if "Final Answer:" in text:
            final_answer = text.split("Final Answer:")[1].strip()
        elif not action_query:
            final_answer = text

        if action_query:
            final_answer = None
                
        past_queries = [action for (_, action, _) in history_list if action]
        if action_query and action_query in past_queries:
            print(f"[{COLOR_YELLOW_BOLD}LOOP PROTECTION{COLOR_RESET}] Repeated query '{action_query}' detected. Forcing real LLM synthesis from full history.")
            
            synth_system_prompt = (
                "You are a Senior Principal Software Architect.\n"
                "You have searched enough. Do not search again. Based on ALL prior retrieved observations below, give your Final Answer now.\n\n"
                "CRITICAL RESPONSE GUIDELINES FOR FINAL ANSWER:\n"
                "1. For QA queries: Output a clear, structured, high-level natural language explanation in plain English. Explain the architecture clearly. DO NOT dump raw function source code bodies.\n"
                "2. For FIX_PROPOSAL queries: Clearly identify the exact target file path, method/function name, line numbers, and technical root cause of the bug.\n"
                "Begin your response with 'Final Answer:' followed by your detailed response."
            )
            
            synth_user_prompt = f"Question: {state['question']}\n\nAll Retrieved Code Observations:\n{history_str}\n\nGive your Final Answer now."
            
            api_synth_start = time.time()
            synth_response = call_with_retry(client, 
                model=model_reasoning,
                messages=[
                    {"role": "system", "content": synth_system_prompt},
                    {"role": "user", "content": synth_user_prompt}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            api_synth_elapsed = time.time() - api_synth_start
            api_elapsed += api_synth_elapsed
            
            synth_text = extract_message_content(synth_response).strip()
            action_query = None
            if "Final Answer:" in synth_text:
                final_answer = synth_text.split("Final Answer:")[1].strip()
            else:
                final_answer = synth_text
            text = f"Thought: Loop Protection triggered -> Forced Synthesis Call\nFinal Answer: {final_answer}"

        if not action_query and not final_answer:
            current_thought = text
            final_answer = text
            
        total_elapsed = time.time() - start_time
        internal_elapsed = max(0.0, total_elapsed - api_elapsed)
        updated_latencies = record_latency(state, "reasoning", total_elapsed, api_elapsed)
        
        print(f"\n--- [{COLOR_CYAN_BOLD}REASONING AGENT{COLOR_RESET}] Iteration {iterations} (Model: {model_reasoning} | API: {api_elapsed:.3f}s | Internal: {internal_elapsed:.3f}s | Total: {total_elapsed:.3f}s) ---")
        print(text)
            
        return {
            "current_thought": current_thought,
            "action_query": action_query,
            "final_answer": final_answer,
            "reasoning_diagnosis": final_answer,
            "iterations": iterations,
            "node_latencies": updated_latencies
        }

    def tool_node(state: AgentState):
        start_time = time.time()
        action = state["action_query"]
        
        api_start = time.time()
        results = hybrid_search(
            query=action,
            collection_name=collection_name,
            persist_dir=persist_dir,
            metadata_index=metadata_index,
            body_index=body_index,
            chunks=bm25_chunks,
            nvidia_api_key=nvidia_api_key,
            top_k=5,
            metadata_weight=3.0,
            body_weight=1.0
        )
        api_elapsed = time.time() - api_start
        
        observation = ""
        for r in results:
            header = f"File: {r['file_path']} | Method/Function: {r['name']} | Lines: {r['start_line']}-{r['end_line']}"
            code_snippet = r['code']
            observation += f"\n--- {header} ---\n{code_snippet}\n"
            
        history = list(state.get("history", []))
        history.append((state.get("current_thought", ""), action, observation))
        
        total_elapsed = time.time() - start_time
        internal_elapsed = max(0.0, total_elapsed - api_elapsed)
        updated_latencies = record_latency(state, "tool", total_elapsed, api_elapsed)
        
        print(f"\n--- [{COLOR_BLUE_BOLD}TOOL NODE (HYBRID RETRIEVAL){COLOR_RESET}] Search: '{action}' (API: {api_elapsed:.3f}s | Internal: {internal_elapsed:.3f}s | Total: {total_elapsed:.3f}s) ---")
        print(f"Retrieved {len(results)} chunks matching '{action}'")
        
        return {
            "history": history,
            "action_query": None,
            "node_latencies": updated_latencies
        }

    def verifier_node(state: AgentState):
        start_time = time.time()
        final_answer = state.get("final_answer", "")
        history_list = state.get("history", [])
        attempts = state.get("verification_attempts", 0) + 1
        
        all_observations = ""
        for step_idx, (thought, action, observation) in enumerate(history_list):
            all_observations += f"\n--- Step {step_idx + 1} (Query: {action}) ---\n{observation}\n"
            
        system_prompt = (
            "You are an unbiased Verification Auditor and Grounding Specialist.\n"
            "Your sole job is to cross-examine a proposed Final Answer against retrieved codebase observations.\n\n"
            "Check if EVERY technical claim, file path, line number, class name, and function behavior in the proposed answer is EXPLICITLY supported by the observations.\n\n"
            "Output EXACTLY one of the following formats:\n\n"
            "FORMAT 1 (If fully grounded):\n"
            "VERDICT: SUPPORTED\n\n"
            "FORMAT 2 (If any claim is unverified/hallucinated):\n"
            "VERDICT: UNSUPPORTED\n"
            "REASON: [Specific bullet points of unsupported claims or missing evidence]\n"
        )
        
        user_prompt = (
            f"Proposed Final Answer:\n{final_answer}\n\n"
            f"Retrieved Codebase Observations:\n{all_observations}\n\n"
            f"Audit the answer for grounding now."
        )
        
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        
        api_start = time.time()
        response = call_with_retry(client, 
            model=model_verifier,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens= 8192
        )
        api_elapsed = time.time() - api_start
        total_elapsed = time.time() - start_time
        internal_elapsed = max(0.0, total_elapsed - api_elapsed)
        
        verifier_text = extract_message_content(response).strip()
        updated_latencies = record_latency(state, "verifier", total_elapsed, api_elapsed)
        
        print(f"\n--- [{COLOR_YELLOW_BOLD}VERIFIER AGENT - Attempt {attempts}{COLOR_RESET}] (Model: {model_verifier} | API: {api_elapsed:.3f}s | Internal: {internal_elapsed:.3f}s | Total: {total_elapsed:.3f}s) ---")
        print(verifier_text)
        
        if "VERDICT: SUPPORTED" in verifier_text:
            return {
                "is_grounded": True,
                "verification_attempts": attempts,
                "node_latencies": updated_latencies
            }
        else:
            return {
                "is_grounded": False,
                "verifier_feedback": verifier_text,
                "verification_attempts": attempts,
                "final_answer": None,
                "node_latencies": updated_latencies
            }

    def fix_proposal_node(state: AgentState):
        start_time = time.time()
        question = state["question"]
        history_list = state.get("history", [])
        verifier_feedback = state.get("verifier_feedback", None)
        reasoning_diagnosis = state.get("reasoning_diagnosis", "")
        
        all_observations = ""
        for step_idx, (thought, action, observation) in enumerate(history_list):
            all_observations += f"\n--- Step {step_idx + 1} (Query: {action}) ---\n{observation}\n"
            
        system_prompt = (
            "You are a Staff Security Engineer and Automated Patch Specialist.\n"
            "Analyze the bug report and retrieved codebase observations to synthesize a production-grade code fix proposal.\n\n"
            "You MUST output your fix proposal using EXACTLY the following key-value format:\n\n"
            "ROOT_CAUSE: [Provide a precise 2-3 sentence technical diagnosis of why the existing code fails]\n\n"
            "FILE_PATH: [exact path]\n"
            "TARGET_START_LINE: [integer]\n"
            "TARGET_END_LINE: [integer]\n"
            "REPLACEMENT_CODE:\n"
            "```python\n"
            "[exact replacement code fixing the bug cleanly]\n"
            "```\n\n"
            "EXPLANATION: [Step-by-step explanation of how the patch resolves the bug safely]\n\n"
            "CRITICAL: The TARGET_START_LINE and TARGET_END_LINE must reference the SAME chunk boundaries provided in the retrieved observations.\n"
            "CRITICAL: Never use '...' or any other elision/truncation to shorten code in "
            "REPLACEMENT_CODE. Reproduce every single line of the target "
            "function in FULL, character-for-character, including unchanged lines.\n\n"
            "CRITICAL: Do NOT output any reasoning, thinking, or conversational text. "
            "Your entire response MUST strictly be the requested key-value format starting directly with ROOT_CAUSE: and ending with EXPLANATION:\n"
        )
        
        if verifier_feedback:
            system_prompt += f"\n\nCRITICAL PREVIOUS TEST FAILURE FEEDBACK FROM VERIFIER AGENT:\n{verifier_feedback}\nFix the issues indicated above.\n"
            
        user_prompt = (
            f"Bug Report: {question}\n\n"
            f"Reasoning Agent's Diagnosis (primary root-cause source — use this as your main basis for FILE_PATH, ROOT_CAUSE, and what code needs to change; the raw observations below are supporting evidence only):\n{reasoning_diagnosis}\n\n"
            f"Retrieved Code Observations:\n{all_observations}\n\n"
            f"Synthesize structured fix proposal now."
        )
        
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        
        api_start = time.time()
        response = call_with_retry(client, 
            model=model_fix_proposal,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=8192
        )
        api_elapsed = time.time() - api_start
        total_elapsed = time.time() - start_time
        internal_elapsed = max(0.0, total_elapsed - api_elapsed)
        
        text = extract_message_content(response).strip()
        updated_latencies = record_latency(state, "fix_proposal", total_elapsed, api_elapsed)
        
        print(f"\n--- [{COLOR_CYAN_BOLD}FIX PROPOSAL AGENT{COLOR_RESET}] (Model: {model_fix_proposal} | API: {api_elapsed:.3f}s | Internal: {internal_elapsed:.3f}s | Total: {total_elapsed:.3f}s) ---")
        print(text)
        
        import re as _re_check
        file_path = "target_file.py"
        root_cause = "Bug detected in target method"
        target_start_line = 1
        target_end_line = 1
        raw_replacement_code = ""
        explanation = ""
        
        try:
            if "FILE_PATH:" in text:
                file_path = text.split("FILE_PATH:")[1].split("\n")[0].strip()
                file_path = file_path.strip("`'\" ")
            if "ROOT_CAUSE:" in text:
                root_cause = text.split("ROOT_CAUSE:")[1].split("FILE_PATH:")[0].strip()
            if "TARGET_START_LINE:" in text:
                target_start_line = int(text.split("TARGET_START_LINE:")[1].split("\n")[0].strip())
            if "TARGET_END_LINE:" in text:
                target_end_line = int(text.split("TARGET_END_LINE:")[1].split("\n")[0].strip())
            if "REPLACEMENT_CODE:" in text:
                repl_part = text.split("REPLACEMENT_CODE:")[1]
                if "EXPLANATION:" in repl_part:
                    repl_code_part = repl_part.split("EXPLANATION:")[0]
                    explanation = repl_part.split("EXPLANATION:")[1].strip()
                else:
                    repl_code_part = repl_part
                raw_replacement_code = repl_code_part.strip()
        except Exception as parse_e:
            pass
            
        cleaned_replacement_code = clean_code_snippet(raw_replacement_code)
        
        truncation_pattern = _re_check.compile(r'^\s*\.\.\.\s*$', _re_check.MULTILINE)
        if truncation_pattern.search(cleaned_replacement_code):
            elapsed_time = time.time() - start_time
            updated_latencies = record_latency(state, "fix_proposal", elapsed_time, 0.0)
            return {
                "proposed_fix": {
                    "file_path": file_path,
                    "root_cause": root_cause,
                    "target_start_line": target_start_line,
                    "target_end_line": target_end_line,
                    "replacement_code": cleaned_replacement_code,
                    "explanation": explanation,
                    "raw_text": text
                },
                "verifier_feedback": "Your REPLACEMENT_CODE contained '...' truncation. You must reproduce the ENTIRE function with zero elisions, every line, character-for-character.",
                "verification_attempts": state.get("verification_attempts", 0) + 1,
                "final_answer": None,
                "node_latencies": updated_latencies
            }
        
        proposed_fix = {
            "file_path": file_path,
            "root_cause": root_cause,
            "target_start_line": target_start_line,
            "target_end_line": target_end_line,
            "replacement_code": cleaned_replacement_code,
            "explanation": explanation,
            "raw_text": text
        }
        
        return {
            "proposed_fix": proposed_fix,
            "final_answer": None,
            "node_latencies": updated_latencies
        }

    def execution_verifier_node(state: AgentState):
        start_time = time.time()
        proposed_fix = state.get("proposed_fix", {})
        attempts = state.get("verification_attempts", 0) + 1
        replacement_code = clean_code_snippet(proposed_fix.get("replacement_code", ""))
        target_start_line = proposed_fix.get("target_start_line", 1)
        target_end_line = proposed_fix.get("target_end_line", 1)
        rel_file_path = proposed_fix.get("file_path", "module.py").strip()
        
        print(f"\n--- [{COLOR_YELLOW_BOLD}EXECUTION VERIFIER AGENT{COLOR_RESET}] Attempt {attempts} (Model: {model_test_generator}) ---")
        
        # 1. Resolve target file path on local filesystem with priority for exact suffix match
        target_full_path = rel_file_path
        if not os.path.exists(target_full_path):
            target_file_name = os.path.basename(rel_file_path.replace("\\", "/"))
            best_candidate = None
            for r_dir, _, files in os.walk("."):
                if target_file_name in files:
                    candidate = os.path.join(r_dir, target_file_name)
                    if candidate.replace("\\", "/").endswith(rel_file_path.replace("\\", "/")):
                        best_candidate = candidate
                        break
                    elif best_candidate is None:
                        best_candidate = candidate
            if best_candidate:
                target_full_path = best_candidate
                        
        full_source_content = ""
        file_found = os.path.exists(target_full_path)
        if file_found:
            try:
                with open(target_full_path, "r", encoding="utf-8", errors="replace") as f:
                    full_source_content = f.read()
            except Exception as e:
                file_found = False
                
        # 2. Generate patched file content using exact line numbers instead of string replacement
        match_found = False
        patched_file_content = ""
        original_error_msg = ""
        
        if file_found and full_source_content:
            try:
                with open(target_full_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # Use 1-indexed to 0-indexed adjustment for list slicing
                # target_end_line is inclusive in the chunk output usually, but python slices are exclusive
                start_idx = max(0, target_start_line - 1)
                end_idx = min(len(lines), target_end_line)
                
                lines[start_idx:end_idx] = [replacement_code + "\n"]
                patched_file_content = "".join(lines)
                match_found = True
                print(f"✓ Line Number Replacement Applied: Lines {target_start_line}-{target_end_line}")
            except Exception as e:
                original_error_msg = f"Failed to patch file by line numbers: {e}"
                print(f"✗ File Patching Failed -> {original_error_msg}")
        else:
            original_error_msg = f"Target file '{rel_file_path}' could not be read on filesystem"
            print(f"✗ File Load Failed -> {original_error_msg}")
            
        # 3. Perform AST Validation on FULL patched file content
        syntax_valid = False
        syntax_error_msg = ""
        
        if match_found and patched_file_content:
            try:
                ast.parse(patched_file_content)
                syntax_valid = True
                print("✓ Full File AST Syntax Check: Passed (Valid Python File Syntax)")
            except SyntaxError as se:
                syntax_error_msg = f"SyntaxError in patched target file: {se}"
                print(f"✗ Full File AST Syntax Check: Failed -> {syntax_error_msg}")
        elif not match_found and original_error_msg:
            syntax_error_msg = original_error_msg
            syntax_valid = False
        else:
            try:
                ast.parse(replacement_code)
                syntax_valid = True
                print("✓ Standalone AST Syntax Check: Passed")
            except SyntaxError as se:
                syntax_error_msg = f"SyntaxError in replacement code: {se}"
                print(f"✗ Standalone AST Syntax Check: Failed -> {syntax_error_msg}")
                
        if not syntax_valid:
            elapsed_time = time.time() - start_time
            updated_latencies = record_latency(state, "execution_verifier", elapsed_time, 0.0)
            
            if attempts >= 7:
                failed_report = (
                    f"[FIX FAILED VERIFICATION]\n\n"
                    f"**Target File**: `{rel_file_path}`\n\n"
                    f"**Root Cause Diagnosis**:\n{proposed_fix.get('root_cause', 'N/A')}\n\n"
                    f"**Last Attempted Replacement Code**:\n```python\n{replacement_code}\n```\n\n"
                    f"**Verification Failure Detail**:\n"
                    f"- **AST Syntax Status**: FAILED ({syntax_error_msg})\n"
                    f"- **Sandbox Unit Test Status**: NOT RUN (Blocked by AST Syntax Error)\n"
                    f"```text\n{syntax_error_msg}\n```"
                )
                return {
                    "verification_attempts": attempts,
                    "test_results": {"passed": False, "output": syntax_error_msg},
                    "final_answer": failed_report,
                    "node_latencies": updated_latencies
                }
            else:
                return {
                    "verification_attempts": attempts,
                    "verifier_feedback": f"Verification Failure: {syntax_error_msg}. Adjust fix proposal accordingly.",
                    "test_results": {"passed": False, "output": syntax_error_msg},
                    "final_answer": None,
                    "node_latencies": updated_latencies
                }
                
        # 4. FAST SINGLE-FILE SANDBOX: Write patched single file & test script to temp directory with PYTHONPATH
        test_gen_prompt = (
            "You are a Senior QA Test Engineer creating an automated unit test script.\n"
            "Given a bug report and a proposed fix, generate a self-contained Python test script (using pytest or unittest)\n"
            "that reproduces the bug and asserts that the replacement fix operates correctly.\n\n"
            "Output ONLY valid Python code inside a single ```python codeblock.\n"
        )
        
        user_test_prompt = (
            f"Bug Report: {state['question']}\n"
            f"Target File: {rel_file_path}\n"
            f"The target file's real module path for imports is: {rel_file_path}\n"
            f"Proposed Replacement Code:\n{replacement_code}\n"
            "Generate unit test script now."
        )
        
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        
        api_start = time.time()
        test_response = call_with_retry(client, 
            model=model_test_generator,
            messages=[
                {"role": "system", "content": test_gen_prompt},
                {"role": "user", "content": user_test_prompt}
            ],
            temperature=0.0,
            max_tokens=2048
        )
        api_elapsed = time.time() - api_start
        
        test_script_text = extract_message_content(test_response).strip()
        test_code = clean_code_snippet(test_script_text)
            
        print("✓ Dynamic Test Case Generator: Script Generated.")
        
        temp_dir = tempfile.mkdtemp(prefix="agent_sandbox_")
        test_passed = False
        exec_output = ""
        
        try:
            sandbox_target_path = os.path.join(temp_dir, rel_file_path)
            os.makedirs(os.path.dirname(sandbox_target_path), exist_ok=True)
            with open(sandbox_target_path, "w", encoding="utf-8") as stf:
                stf.write(patched_file_content if patched_file_content else replacement_code)
                
            test_file_path = os.path.join(temp_dir, "test_bug_fix.py")
            with open(test_file_path, "w", encoding="utf-8") as tf:
                tf.write(test_code)
                
            env = os.environ.copy()
            repo_root_abs = os.path.abspath(".")
            env["PYTHONPATH"] = f"{temp_dir}{os.pathsep}{repo_root_abs}{os.pathsep}" + env.get("PYTHONPATH", "")
            
            res = subprocess.run(
                [sys.executable, "-m", "pytest", test_file_path],
                capture_output=True,
                text=True,
                timeout=150,
                cwd=temp_dir,
                env=env
            )
            
            exec_output = res.stdout + "\n" + res.stderr
            if res.returncode == 0 or "passed" in res.stdout.lower():
                test_passed = True
                print("✓ Sandboxed Test Execution Engine: PASSED (Bug Resolution Confirmed)")
            else:
                print(f"! Sandboxed Test Execution Engine: FAILED (Exit Code {res.returncode})")
        except Exception as e:
            exec_output = f"Execution error: {str(e)}"
            print(f"! Sandbox Execution Exception: {str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        total_elapsed = time.time() - start_time
        internal_elapsed = max(0.0, total_elapsed - api_elapsed)
        updated_latencies = record_latency(state, "execution_verifier", total_elapsed, api_elapsed)
        
        print(f"--- [{COLOR_YELLOW_BOLD}EXECUTION VERIFIER COMPLETED{COLOR_RESET}] (API: {api_elapsed:.3f}s | Internal: {internal_elapsed:.3f}s | Total: {total_elapsed:.3f}s) ---")
        
        if test_passed:
            final_patch_report = (
                f"### Category 2 Bug Fix Proposal (Verified)\n\n"
                f"**Target File**: `{rel_file_path}`\n\n"
                f"**Root Cause Diagnosis**:\n{proposed_fix.get('root_cause', 'N/A')}\n\n"
                f"**Target Lines Edited**: {target_start_line}-{target_end_line}\n\n"
                f"**Proposed Replacement Code**:\n```python\n{replacement_code}\n```\n\n"
                f"**Explanation**:\n{proposed_fix.get('explanation', 'Bug fix applied cleanly.')}\n\n"
                f"--- \n### Dynamic Execution Verification\n"
                f"- **Full File AST Syntax Status**: Passed\n"
                f"- **Sandbox Unit Test Status**: PASSED (Confirmed Bug Resolved)\n"
                f"```text\n{exec_output[:500] if exec_output else 'Executed successfully.'}\n```"
            )
            return {
                "verification_attempts": attempts,
                "sandbox_test_script": test_code,
                "test_results": {"passed": True, "output": exec_output},
                "final_answer": final_patch_report,
                "node_latencies": updated_latencies
            }
        else:
            if attempts >= 7:
                failed_report = (
                    f"[FIX FAILED VERIFICATION]\n\n"
                    f"**Target File**: `{rel_file_path}`\n\n"
                    f"**Root Cause Diagnosis**:\n{proposed_fix.get('root_cause', 'N/A')}\n\n"
                    f"**Last Attempted Replacement Code**:\n```python\n{replacement_code}\n```\n\n"
                    f"**Verification Failure Detail**:\n"
                    f"- **Full File AST Syntax Status**: Passed\n"
                    f"- **Sandbox Unit Test Status**: FAILED\n"
                    f"```text\n{exec_output[:500] if exec_output else 'Test execution failed.'}\n```"
                )
                return {
                    "verification_attempts": attempts,
                    "sandbox_test_script": test_code,
                    "test_results": {"passed": False, "output": exec_output},
                    "final_answer": failed_report,
                    "node_latencies": updated_latencies
                }
            else:
                return {
                    "verification_attempts": attempts,
                    "verifier_feedback": f"Sandbox Test Execution Failed:\n{exec_output[:400]}\nAdjust the replacement_code to resolve the test failure.",
                    "test_results": {"passed": False, "output": exec_output},
                    "final_answer": None,
                    "node_latencies": updated_latencies
                }

    # Graph Routers
    def route_reasoning(state: AgentState):
        # 1. If an action_query is present, ALWAYS route to tool_node first!
        if state.get("action_query") is not None:
            return "tool"
            
        # 2. Only route to verifier / fix_proposal after all searches are done and final_answer exists
        if state.get("final_answer") is not None or state.get("iterations", 0) >= 15:
            if state.get("intent_category") == "FIX_PROPOSAL":
                return "fix_proposal"
            else:
                return "verifier"
                
        return "tool"

    def route_qa_verification(state: AgentState):
        if state.get("is_grounded", False) or state.get("verification_attempts", 0) >= 2:
            return "end"
        if state.get("final_answer") is not None:
            return "end"
        return "re_reason"

    def route_fix_verification(state: AgentState):
        if state.get("final_answer") is not None:
            return "end"
        return "re_fix"

    # Assemble Workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("fix_proposal", fix_proposal_node)
    workflow.add_node("execution_verifier", execution_verifier_node)
    
    workflow.set_entry_point("intent_classifier")
    workflow.add_edge("intent_classifier", "reasoning")

    workflow.add_conditional_edges(
        "reasoning",
        route_reasoning,
        {
            "verifier": "verifier",
            "fix_proposal": "fix_proposal",
            "tool": "tool"
        }
    )
    
    workflow.add_conditional_edges(
        "verifier",
        route_qa_verification,
        {
            "end": END,
            "re_reason": "reasoning"
        }
    )

    workflow.add_conditional_edges(
        "execution_verifier",
        route_fix_verification,
        {
            "end": END,
            "re_fix": "fix_proposal"
        }
    )

    workflow.add_edge("tool", "reasoning")
    workflow.add_edge("fix_proposal", "execution_verifier")
    
    app = workflow.compile()
    app.configured_models = configured_models
    return app


# Compile Dual-Mode LangGraph Agent Graph
print("Compiling Dual-Mode LangGraph Agent...")
agent_app = build_agent_graph(
    collection_name=collection_name,
    persist_dir=persist_dir,
    metadata_index=metadata_index,
    body_index=body_index,
    bm25_chunks=bm25_chunks,
    nvidia_api_key=nvidia_api_key
)
print("Dual-Mode Agent graph compiled successfully.\n")

test_queries = [
    # "what LLM and it's framework are we using in it?",
    "In dataflow.py inside def execute_tool function, there is a bug with sanitization where it marks sanitization = False for all vulnerabilities. Propose a fix for this bug and verify it."
]

print("Testing Dual-Mode Agent Pipeline:\n")

for query in test_queries:
    print("=" * 90)
    print(f"USER QUERY: '{query}'")
    print("=" * 90)
    
    initial_state = {
        "question": query,
        "intent_category": None,
        "history": [],
        "current_thought": "",
        "action_query": None,
        "final_answer": None,
        "proposed_fix": None,
        "sandbox_test_script": None,
        "test_results": None,
        "iterations": 0,
        "verification_attempts": 0,
        "verifier_feedback": None,
        "is_grounded": False,
        "node_latencies": {}
    }
    
    final_state = agent_app.invoke(initial_state)
    
    print("\n" + "-" * 90)
    if "[WARNING: Partially Grounded]" in (final_state.get("final_answer") or ""):
        print(f"{COLOR_YELLOW_BOLD}FINAL AGENT OUTPUT (GROUNDING CHECK FAILED — SEE WARNING BELOW):{COLOR_RESET}")
    else:
        print(f"{COLOR_GREEN_BOLD}FINAL VERIFIED AGENT OUTPUT:{COLOR_RESET}")
    print(f"{COLOR_GREEN}{final_state.get('final_answer')}{COLOR_RESET}")
    print("-" * 90)
    
    # Print colored latency benchmark table
    print_latency_benchmark_report(final_state, agent_app.configured_models)


