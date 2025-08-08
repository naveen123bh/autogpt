import json
import requests
from duckduckgo_search import DDGS
import faiss
import numpy as np

OLLAMA_MODEL = "mistral"  # or any model you've started with `ollama run ...`
EMBED_DIM = 384
MAX_STEPS = 5

# --- Vector memory store ---
memory_index = faiss.IndexFlatL2(EMBED_DIM)
memory_texts = []

def embed_text(text):
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(EMBED_DIM).astype("float32")

def add_memory(text):
    vec = embed_text(text)
    memory_index.add(vec.reshape(1, -1))
    memory_texts.append(text)

def recall_memory(query, k=3):
    if not memory_texts:
        return []
    qvec = embed_text(query).reshape(1, -1)
    D, I = memory_index.search(qvec, min(k, len(memory_texts)))
    return [memory_texts[i] for i in I[0]]

# --- Tools ---
def tool_search(query):
    print(f"[Search] {query}")
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            results.append(f"{r['title']}: {r['body']} ({r['href']})")
    return "\n".join(results) if results else "No results found."

def tool_write_file(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved to {filename}"

tools = {
    "search": tool_search,
    "write_file": tool_write_file,
}

# --- Talk to Ollama ---
def call_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )
    return response.json()["response"].strip()

# --- Plan next step ---
def plan_step(goal, recent_mem):
    prompt = f"""
You are an autonomous AI agent.
Goal: {goal}
Recent memory: {recent_mem}
Available tools: search(query), write_file(filename:::content)
Return a JSON like:
{{"action":"search","input":"..."}}
OR
{{"action":"write_file","input":"file.txt:::file content"}}
OR
{{"action":"finish","input":""}}
"""
    text = call_ollama(prompt)
    try:
        return json.loads(text)
    except:
        return {"action": "finish", "input": ""}

# --- Main loop ---
def run_agent(goal):
    for step in range(MAX_STEPS):
        print(f"\n--- Step {step+1} ---")
        mem = recall_memory(goal)
        action = plan_step(goal, mem)

        if action["action"] == "finish":
            print("✅ Goal complete.")
            break
        elif action["action"] in tools:
            if action["action"] == "write_file":
                if ":::" in action["input"]:
                    name, content = action["input"].split(":::", 1)
                    result = tool_write_file(name.strip(), content.strip())
                else:
                    result = "Invalid format"
            else:
                result = tools[action["action"]](action["input"])
            add_memory(f"{action['action']} -> {result}")
            print(result)
        else:
            print(f"⚠️ Unknown action: {action}")
            break

if __name__ == "__main__":
    goal = input("Enter your goal: ")
    run_agent(goal)
