"""
How this works:
This script indexes all Python files in a codebase directory and stores them in a vector database for semantic search.
1. First, we recursively walk the target codebase folder to find all files ending with '.py'.
2. We read each file and use the Tree-sitter library to parse the syntax of the code. We look at the top-level
   elements of the code to find function and class definitions and extract their code as separate text chunks.
3. Next, we call the NVIDIA NIM API using the 'nvidia/llama-nemotron-embed-1b-v2' model. This converts our code
   chunks into high-dimensional embeddings (lists of numbers representing semantic meaning). We chose this model
   because standard embedding models have a very short token limit and would truncate long code blocks. Using this
   Llama model allows us to handle very long context windows, ensuring no code is lost.
4. We store these embeddings and their associated metadata (file path, class name, name, type, lines) in a Chroma vector database.
5. We also build a sparse BM25 keyword index over the codebase chunks using the rank_bm25 package.
6. When querying, we perform a Hybrid Search: we retrieve the top 20 matches from the dense embedding database and the
   top 20 matches from the BM25 keyword search, and merge them using Reciprocal Rank Fusion (RRF).
7. To answer complex queries, we build a ReAct (Reasoning and Action) agent loop using LangGraph.
   - The Agent starts at the reasoning node, prompting Meta's Llama 3.3 model on NVIDIA NIM to think about what it needs
     and decide whether to perform a search or declare a final answer.
   - If it decides to search, it runs the hybrid_search tool, appends the code observation to its history, and loops back.
   - If it hits the cap of 5 iterations without finding an answer, it forces a partial answer summary.
   - All steps (Thoughts, Actions, and observations) are logged to the console for debugging.
"""

import os
import sys
import uuid
import chromadb
import tree_sitter
import tree_sitter_python
from openai import OpenAI
from rank_bm25 import BM25Okapi
from typing import TypedDict, List, Tuple, Optional
from langgraph.graph import StateGraph, END

# --- Agent State Definition ---
class AgentState(TypedDict):
    question: str
    history: List[Tuple[str, str, str]]  # list of (thought, action, observation)
    current_thought: str
    action_query: Optional[str]
    final_answer: Optional[str]
    iterations: int

# --- AST Chunker Functions ---
def chunk_repo(repo_path):
    """
    Recursively scans the directory specified by repo_path for Python (.py) files,
    parses each file using Tree-sitter, and extracts top-level function definitions,
    class methods, and non-trivial class-level statements (like docstrings and properties)
    as separate chunks.
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
                            start_line = child.start_point[0] + 1
                            end_line = child.end_point[0] + 1
                            
                            chunks.append({
                                "code": code_text,
                                "file_path": rel_path,
                                "class_name": "",
                                "name": name,
                                "type": "function",
                                "start_line": start_line,
                                "end_line": end_line
                            })
                        
                        elif child.type == 'class_definition':
                            class_name_node = child.child_by_field_name('name')
                            class_name = class_name_node.text.decode('utf-8', errors='replace') if class_name_node else 'unknown'
                            
                            class_block = None
                            for c in child.children:
                                if c.type == 'block':
                                    class_block = c
                                    break
                            
                            if class_block is not None:
                                for member in class_block.children:
                                    if member.type == 'function_definition':
                                        method_name_node = member.child_by_field_name('name')
                                        method_name = method_name_node.text.decode('utf-8', errors='replace') if method_name_node else 'unknown'
                                        
                                        method_code = content_bytes[member.start_byte:member.end_byte].decode('utf-8', errors='replace')
                                        m_start = member.start_point[0] + 1
                                        m_end = member.end_point[0] + 1
                                        
                                        chunks.append({
                                            "code": method_code,
                                            "file_path": rel_path,
                                            "class_name": class_name,
                                            "name": method_name,
                                            "type": "method",
                                            "start_line": m_start,
                                            "end_line": m_end
                                        })
                                    else:
                                        member_text = content_bytes[member.start_byte:member.end_byte].decode('utf-8', errors='replace').strip()
                                        if member_text and member_text not in (';', 'pass') and len(member_text) > 5:
                                            m_start = member.start_point[0] + 1
                                            m_end = member.end_point[0] + 1
                                            
                                            chunks.append({
                                                "code": member_text,
                                                "file_path": rel_path,
                                                "class_name": class_name,
                                                "name": class_name + "_definition",
                                                "type": "class_definition",
                                                "start_line": m_start,
                                                "end_line": m_end
                                            })
                except Exception as error:
                    print("Warning: Failed to parse file", rel_path, ":", error)
                    continue
                    
    return chunks

# --- Vector Storage Functions ---
def embed_and_store(chunks, collection_name, persist_dir, nvidia_api_key):
    """
    Converts code chunks into mathematical embeddings using NVIDIA NIM API (llama-nemotron-embed-1b-v2)
    and stores those embeddings along with their metadata inside a local, persistent Chroma vector database.
    """
    if not chunks:
        print("No chunks to embed and store.")
        return

    print("Initializing NVIDIA NIM API connection...")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_api_key
    )

    print("Connecting to persistent Chroma database at:", persist_dir)
    client_db = chromadb.PersistentClient(path=persist_dir)

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
            if chunk['type'] == 'method':
                text_to_embed = "File: " + chunk['file_path'] + "\nClass: " + chunk['class_name'] + "\nMethod: " + chunk['name'] + "\nType: method\n\n" + chunk['code']
            elif chunk['type'] == 'class_definition':
                text_to_embed = "File: " + chunk['file_path'] + "\nClass: " + chunk['class_name'] + "\nType: class_definition\n\n" + chunk['code']
            else:
                text_to_embed = "File: " + chunk['file_path'] + "\nName: " + chunk['name'] + "\nType: function\n\n" + chunk['code']
            batch_texts_to_embed.append(text_to_embed)

        response = client.embeddings.create(
            input=batch_texts_to_embed,
            model="nvidia/llama-nemotron-embed-1b-v2",
            extra_body={"input_type": "passage", "truncate": "NONE"}
        )
        
        batch_embeddings = []
        for data in response.data:
            batch_embeddings.append(data.embedding)

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

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_codes
        )

# --- Dense Search ---
def query_repo(query, collection_name, persist_dir, nvidia_api_key, top_k=5):
    """
    Converts a natural language query into a vector embedding using NVIDIA NIM and searches the
    Chroma vector database to find the top_k most semantically similar code chunks.
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_api_key
    )
    
    response = client.embeddings.create(
        input=query,
        model="nvidia/llama-nemotron-embed-1b-v2",
        extra_body={"input_type": "query"}
    )
    query_embedding = response.data[0].embedding

    client_db = chromadb.PersistentClient(path=persist_dir)

    try:
        collection = client_db.get_collection(name=collection_name)
    except Exception as error:
        print("Error: Collection '", collection_name, "' not found:", error)
        return []

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

                for idx in range(len(docs)):
                    document_text = docs[idx]
                    metadata = metadatas[idx]
                    distance = distances[idx]
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

# --- Sparse Search Helper Functions ---
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
    Constructs two separate BM25Okapi search indexes over the codebase chunks:
    one for metadata and one for the code body.
    """
    metadata_corpus = []
    body_corpus = []
    for chunk in chunks:
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

# --- Hybrid Search using Reciprocal Rank Fusion ---
def hybrid_search(query, collection_name, persist_dir, metadata_index, body_index, chunks, nvidia_api_key, top_k=5, rrf_k=60, metadata_weight=3.0, body_weight=1.0):
    """
    Fuses dense embedding similarity and field-weighted sparse BM25 keyword matching scores using Reciprocal Rank Fusion.
    """
    dense_results = query_repo(query, collection_name, persist_dir, nvidia_api_key, top_k=20)
    
    bm25_results = bm25_search(
        query=query,
        metadata_index=metadata_index,
        body_index=body_index,
        chunks=chunks,
        top_k=20,
        metadata_weight=metadata_weight,
        body_weight=body_weight
    )
    
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
        
    all_keys = set(dense_rank.keys()).union(set(bm25_rank.keys()))
    
    fused_results = []
    for key in all_keys:
        d_rank = dense_rank.get(key)
        rrf_dense = 1.0 / (rrf_k + d_rank) if d_rank is not None else 0.0
            
        b_rank = bm25_rank.get(key)
        rrf_bm25 = 1.0 / (rrf_k + b_rank) if b_rank is not None else 0.0
            
        rrf_score = rrf_dense + rrf_bm25
        
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

# --- ReAct Agent Nodes and Flow via LangGraph ---
def build_agent_graph(collection_name, persist_dir, metadata_index, body_index, bm25_chunks, nvidia_api_key):
    """
    Creates and compiles a LangGraph StateGraph that defines the ReAct loop.
    """
    
    def reasoning_node(state: AgentState):
        """
        Prompts Meta's Llama 3.3 model on NVIDIA NIM to reason about the codebase question.
        Generates either an Action (search query) or a Final Answer.
        """
        current_iterations = state.get("iterations", 0)
        history_list = state.get("history", [])
        
        history_str = ""
        for idx, (thought, action, observation) in enumerate(history_list):
            history_str += f"\n--- Step {idx+1} ---\n"
            history_str += f"Thought: {thought}\n"
            history_str += f"Action: {action}\n"
            history_str += f"Observation:\n{observation}\n"

        # FIXED: Since current_iterations is 0-indexed, the 5th run is current_iterations = 4.
        # We must force the LLM to output its Final Answer on this 5th step.
        if current_iterations >= 4:
            system_prompt = (
                "You have reached the maximum search limit. You must summarize the best partial answer "
                "using ONLY the information gathered in the observations so far.\n"
                "You must format your output exactly as:\n"
                "Thought: [summarize findings]\n"
                "Final Answer: Incomplete — max iterations reached. [your answer based on available observations]"
            )
        else:
            system_prompt = (
                "You are an expert codebase QA agent. You have access to a hybrid search tool to locate "
                "relevant code chunks (functions and classes) from the codebase.\n\n"
                "Your task is to answer the user's question about the codebase.\n"
                "For each turn, you can either perform a search or provide a final answer.\n\n"
                "Format requirements:\n"
                "- If you need to search the codebase, output exactly:\n"
                "Thought: [describe what you need to look up and why]\n"
                "Action: [your query string for search]\n\n"
                "- If you have gathered enough information to answer the question, output exactly:\n"
                "Thought: [summarize your findings]\n"
                "Final Answer: [your complete answer based on the retrieved code chunks]\n\n"
                "Do not include any other text outside these fields."
            )
        
        user_prompt = f"Original Question: {state['question']}\n\nSearch History:\n{history_str}\n\nPlease take the next step."
        
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        
        response = client.chat.completions.create(
            model="z-ai/glm-5.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=16384
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Strip markdown code block wrappers if GLM wraps the output
        if response_text.startswith("```"):
            lines = response_text.splitlines()
            cleaned_lines = [l for l in lines if not l.strip().startswith("```")]
            response_text = "\n".join(cleaned_lines).strip()
            
        thought = ""
        action_query = None
        final_answer = None
        
        # Parse Thought and Action/Final Answer
        if "Thought:" in response_text:
            parts = response_text.split("Thought:", 1)
            rest = parts[1]
            if "Action:" in rest:
                thought_part, action_part = rest.split("Action:", 1)
                thought = thought_part.strip()
                action_query = action_part.strip()
            elif "Final Answer:" in rest:
                thought_part, answer_part = rest.split("Final Answer:", 1)
                thought = thought_part.strip()
                final_answer = answer_part.strip()
            else:
                thought = rest.strip()
        else:
            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:", 1)[1].strip()
            elif "Action:" in response_text:
                action_query = response_text.split("Action:", 1)[1].strip()
            else:
                final_answer = response_text

        # Strip residual quotes or backticks from keys
        if action_query:
            action_query = action_query.strip("`'\" \n")
        if final_answer:
            final_answer = final_answer.strip("` \n")

        print(f"\n--- [AGENT REASONING] Iteration {current_iterations + 1} ---")
        print(f"Thought: {thought}")
        if final_answer:
            print(f"Final Answer: {final_answer}")
        else:
            print(f"Action: {action_query}")
            
        return {
            "current_thought": thought,
            "action_query": action_query,
            "final_answer": final_answer,
            "iterations": current_iterations + 1
        }

    def tool_node(state: AgentState):
        """
        Executes the hybrid search tool using the query generated by the reasoning node.
        Appends the resulting code block matches as an observation in the history.
        """
        action = state["action_query"]
        
        # Run hybrid search
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
        
        # Format observations into readable context
        observation = ""
        for r in results:
            observation += f"File: {r['file_path']} | Name: {r['name']} | Type: {r['type']}\n"
            observation += f"Code:\n{r['code']}\n"
            observation += "-" * 40 + "\n"
            
        if not results:
            observation = "No matching code chunks found."
            
        new_history = list(state["history"])
        new_history.append((state["current_thought"], action, observation))
        
        print(f"\n--- [TOOL CALL] ---")
        print(f"Query: '{action}' -> Retrieved {len(results)} chunks.")
        
        return {
            "history": new_history,
            "action_query": None
        }

    # Define Graph workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("tool", tool_node)
    workflow.set_entry_point("reasoning")

    # Routing logic
    def route_agent(state: AgentState):
        if state["final_answer"] is not None:
            return "end"
        if state["iterations"] >= 5:
            return "end"
        return "tool"

    workflow.add_conditional_edges(
        "reasoning",
        route_agent,
        {
            "end": END,
            "tool": "tool"
        }
    )
    workflow.add_edge("tool", "reasoning")
    
    return workflow.compile()

# --- Main CLI Execution ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python repo_indexer.py <repo_path>")
        sys.exit(1)

    repo_path = sys.argv[1]
    if not os.path.isdir(repo_path):
        print("Error: Path '", repo_path, "' is not a directory.")
        sys.exit(1)

    nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not nvidia_api_key:
        print("Error: NVIDIA_API_KEY environment variable is not set.")
        sys.exit(1)

    collection_name = "repo_code_chunks"
    persist_dir = "./chroma_db"

    # Step 1: Chunk repository
    print("Step 1: Chunking codebase inside:", repo_path)
    chunks = chunk_repo(repo_path)
    print("Successfully extracted", len(chunks), "code chunks.")

    if not chunks:
        print("No Python codebase chunks extracted. Exiting.")
        sys.exit(0)

    # Step 2: Embed and store
    print("\nStep 2: Embedding chunks and storing in Chroma using NVIDIA NIM...")
    embed_and_store(chunks, collection_name, persist_dir, nvidia_api_key)

    # Step 3: Compile BM25 indexes (metadata and body)
    print("\nStep 3: Compiling BM25 keyword indexes...")
    metadata_index, body_index, bm25_chunks = build_bm25_index(chunks)
    print("BM25 indexes built successfully.")

    # Step 4: Compile LangGraph Agent
    print("\nStep 4: Compiling ReAct Agent Graph...")
    agent_app = build_agent_graph(
        collection_name=collection_name,
        persist_dir=persist_dir,
        metadata_index=metadata_index,
        body_index=body_index,
        bm25_chunks=bm25_chunks,
        nvidia_api_key=nvidia_api_key
    )
    print("Agent graph compiled successfully.")

    # Step 5: Test Agent Questions
    agent_questions = [
        "What does layer 1 secrets detection check for?",
        "How does the system decide if a finding is a false positive?",
        "What is the overall flow from scanning a file to producing a report, and which file orchestrates it?"
    ]

    print("\nStep 5: Testing ReAct Agent Loop:")
    for question in agent_questions:
        print("\n" + "=" * 90)
        print(f"QUESTION: '{question}'")
        print("=" * 90)
        
        initial_state = {
            "question": question,
            "history": [],
            "current_thought": "",
            "action_query": None,
            "final_answer": None,
            "iterations": 0
        }
        
        final_state = agent_app.invoke(initial_state)
        print("\n" + "-" * 50)
        print("FINAL AGENT ANSWER:")
        print(final_state["final_answer"])
        print("-" * 50)
