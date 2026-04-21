from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

SYSTEM = """You are CellScan AI Assistant for a malaria blood cell detection project. Key facts:
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
    prompt = f"{SYSTEM}\n\nUser: {req.message}\nAssistant:"
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 150, "temperature": 0.7, "return_full_text": False},
                "options": {"wait_for_model": True, "use_cache": False}
            }
        )
        try:
            data = res.json()
            if isinstance(data, list) and data:
                reply = data[0].get("generated_text", "").strip()
                if reply:
                    return {"reply": reply}
        except Exception:
            pass
    return {"reply": "Sorry, try again in a moment."}

@app.get("/")
def root():
    return {"status": "ok"}                headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                json={
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": 150, "temperature": 0.7},
                    "options": {"wait_for_model": True}
                }
            )
            try:
                data = res.json()
                if isinstance(data, list) and data:
                    reply = data[0].get("generated_text", "").split("[/INST]")[-1].strip()
                    if reply:
                        return {"reply": reply}
            except Exception:
                pass
    return {"reply": "Model is warming up, please try again in a moment."}

@app.get("/")
def root():
    return {"status": "ok"}
