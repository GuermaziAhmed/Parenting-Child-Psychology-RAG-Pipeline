"""RAG pipeline logic: retrieve relevant chunks then ask LLM.
"""
from __future__ import annotations
from typing import Dict, Any, List
import chromadb
from config import CHROMA_DIR, CHROMA_COLLECTION, DEFAULT_TOP_K, PROMPT_TEMPLATE
from llm import LLMClient


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(CHROMA_COLLECTION)


def retrieve(query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    collection = get_collection()
    # Using Chroma similarity search
    results = collection.query(query_texts=[query], n_results=k)
    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown")
        })
    return docs


def build_prompt(question: str, docs: List[Dict[str, Any]]) -> str:
    sources_formatted = []
    for d in docs:
        sources_formatted.append(f"[ID: {d['id']}]\n{d['text'][:1000]}")  # truncate for prompt length
    sources_block = "\n\n".join(sources_formatted)
    return PROMPT_TEMPLATE.format(question=question, sources=sources_block)


def answer(question: str, k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    docs = retrieve(question, k=k)
    prompt = build_prompt(question, docs)
    llm = LLMClient()
    response_text = llm.chat(prompt)
    return {
        "question": question,
        "chunks": docs,
        "answer": response_text,
        "prompt": prompt,
    }

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "How can I support my child's emotional development?"
    result = answer(q)
    print("QUESTION:\n", result["question"])    
    print("\nANSWER:\n", result["answer"])    
    print("\nSOURCES USED:")
    for c in result["chunks"]:
        print(f"- {c['id']} (source={c['source']})")
