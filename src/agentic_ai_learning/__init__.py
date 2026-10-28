import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

# Initialize NVIDIA model
llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.3,
    max_completion_tokens=200,
)

# Ask a question
prompt = "Explain Agentic AI in one simple line."

try:
    response = llm.invoke(prompt)
    print("AI Response:")
    print(response.content)
except Exception as e:
    print("Error:", e)