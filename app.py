from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
print("TOKEN:", HF_TOKEN)
print("LENGTH:", len(HF_TOKEN))
SYSTEM = "You are CellScan AI Assistant for a malaria detection project. ResNet18 model, NIH dataset 27558 images, 96.78% accuracy, PyTorch. Answer under 80 words."

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": req.message}
                ],
                "max_tokens": 120,
                "temperature": 0.7
            }
        )

        data = res.json()
        print(data)

        try:
            reply = data["choices"][0]["message"]["content"]
            return {"reply": reply}
        except:
            return {"reply": str(data)}                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": req.message}
                ],
                "max_tokens": 120,
                "temperature": 0.7
            }
        )

        data = res.json()
        print(data)

        try:
            reply = data["choices"][0]["message"]["content"]
            return {"reply": reply}
        except:
            return {"reply": str(data)}
