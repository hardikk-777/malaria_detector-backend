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

SYSTEM = "You are CellScan AI Assistant for a malaria detection project. ResNet18 model, NIH dataset 27558 images, 96.78% accuracy, PyTorch. Answer under 80 words."

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    prompt = SYSTEM + "\n\nUser: " + req.message + "\nAssistant:"
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            headers={"Authorization": "Bearer " + HF_TOKEN, "Content-Type": "application/json"},
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
    return {"status": "ok"}
