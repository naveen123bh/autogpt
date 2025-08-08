# tools.py

from ddgs import DDGS

# Global tool registry
TOOL_REGISTRY = {}

# Tool decorator
def tool(name):
    def wrapper(func):
        TOOL_REGISTRY[name] = func
        return func
    return wrapper

@tool("search")
def search_tool(query):
    print(f"[Tool: search] {query}")
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            results.append(f"{r['title']}: {r['body']} ({r['href']})")
    return "\n".join(results) if results else "No results found."

@tool("write_file")
def write_file_tool(params):
    if ":::" not in params:
        return "Invalid input format. Use filename:::content"
    filename, content = params.split(":::", 1)
    with open(filename.strip(), "w", encoding="utf-8") as f:
        f.write(content.strip())
    return f"Saved to {filename.strip()}"
