from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

SYSTEM = """You are CellScan AI Assistant. This is a malaria blood cell detection project. Key facts:
- Model: ResNet18 pretrained on ImageNet, fine-tuned for Parasitized vs Uninfected classification
- Dataset: NIH Malaria Cell Images, 27,558 images, 70/15/15 split
- Input: 224x224, ImageNet normalization
- Optimizer: Adam, lr=0.001, CrossEntropyLoss, 10 epochs
- Val accuracy: ~96.78%
- Framework: PyTorch, deployed on Hugging Face Spaces with Gradio
Answer concisely under 80 words."""

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    prompt = f"<s>[INST] {SYSTEM}\n\nUser: {req.message} [/INST]"
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.7}}
        )
        data = res.json()
        if isinstance(data, list) and data:
            reply = data[0].get("generated_text", "").split("[/INST]")[-1].strip()
        else:
            reply = "Sorry, the model is loading. Please try again in a few seconds."
    return {"reply": reply}

@app.get("/")
def root():
    return {"status": "ok"}
