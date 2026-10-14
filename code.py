
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load API credentials from .env
load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError("TAVILY_API_KEY not found. Check your .env file.")

# Connect to Tavily
client = TavilyClient(api_key=api_key)

# Run a real web search
query = "What are the latest research findings on AI hallucinations in academic research?"

response = client.search(
    query=query,
    search_depth="basic",
    max_results=5
)

print(f"\nResearch question: {query}\n")

for i, result in enumerate(response["results"], start=1):
    print(f"{i}. {result['title']}")
    print(f"   URL: {result['url']}")
    print(f"   Content: {result['content'][:400]}")
    print()
