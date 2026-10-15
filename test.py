import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("NEBIUS_API_KEY")

if not api_key:
    raise ValueError("NEBIUS_API_KEY not found in .env")

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    messages=[
        {
            "role": "system",
            "content": "You are a careful research assistant. "
                       "Distinguish verified facts from uncertainty."
        },
        {
            "role": "user",
            "content": (
                "In three bullet points, explain why AI hallucinations "
                "are a problem in academic research. Do not invent "
                "research papers or citations."
            )
        }
    ],
    max_tokens=250,
    temperature=0.2,
)

print(response.choices[0].message.content)
