import json
import requests
import numpy as np
from tools import TOOL_REGISTRY
from memory import create_index, save_index, load_index

OLLAMA_MODEL = "mistral"
EMBED_DIM = 384
MAX_STEPS = 5

# --- Persistent vector memory ---
memory_index, memory_texts = load_index()

def embed_text(text):
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(EMBED_DIM).astype("float32")

def add_memory(text):
    vec = embed_text(text)
    memory_index.add(vec.reshape(1, -1))
    memory_texts.append(text)
    save_index(memory_index, memory_texts)

def recall_memory(query, k=3):
    if not memory_texts:
        return []
    qvec = embed_text(query).reshape(1, -1)
    D, I = memory_index.search(qvec, min(k, len(memory_texts)))
    return [memory_texts[i] for i in I[0]]

# --- LLM via Ollama ---
def call_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )
    return response.json()["response"].strip()

# --- Planner ---
def plan_step(goal, recent_mem):
    prompt = f"""
You are an autonomous AI agent.
Goal: {goal}
Recent memory: {recent_mem}
Available tools: {', '.join(TOOL_REGISTRY.keys())}
Return ONLY ONE JSON object like:
{{"action":"tool_name","input":"parameters"}}
OR
{{"action":"finish","input":""}}
No extra text, no lists, no multiple JSON objects.

Examples:
{{"action":"search","input":"Nikola Tesla facts"}}
{{"action":"write_file","input":"tesla.txt:::Fact 1\\nFact 2\\nFact 3"}}
{{"action":"finish","input":""}}
"""
    text = call_ollama(prompt)
    print(f"[Planner output]: {text}")  # Debug print
    try:
        return json.loads(text)
    except Exception as e:
        print(f"[Planner JSON parse error]: {e}")
        return {"action": "finish", "input": ""}

# --- Manual fallback test for Tesla facts ---
def manual_tesla_facts():
    print("[Manual] Searching for Nikola Tesla facts...")
    facts_text = TOOL_REGISTRY["search"]("Nikola Tesla facts")
    print("[Manual] Writing facts to tesla.txt...")
    save_result = TOOL_REGISTRY["write_file"](f"tesla.txt:::{facts_text}")
    print(save_result)

# --- Main loop ---
def run_agent(goal):
    for step in range(MAX_STEPS):
        print(f"\n--- Step {step+1} ---")
        mem = recall_memory(goal)
        action = plan_step(goal, mem)

        if action["action"] == "finish":
            print("✅ Goal complete.")
            break
        elif action["action"] in TOOL_REGISTRY:
            result = TOOL_REGISTRY[action["action"]](action["input"])
            print(result)
            add_memory(f"{action['action']} -> {result}")
        else:
            print(f"⚠️ Unknown action: {action}")
            break

if __name__ == "__main__":
    goal = input("Enter your goal (or type 'manual' to test Tesla facts): ")
    if goal.lower() == "manual":
        manual_tesla_facts()
    else:
        run_agent(goal)
