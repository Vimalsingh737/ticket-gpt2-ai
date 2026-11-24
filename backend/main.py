import json
import os
import re
import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tokenizers import Tokenizer

app = FastAPI(title="TicketGPT-2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ticket_gpt2.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.json")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

BOS_ID = tokenizer.token_to_id("[BOS]")
MSG_ID = tokenizer.token_to_id("[MSG]")
JSON_ID = tokenizer.token_to_id("[JSON]")
EOS_ID = tokenizer.token_to_id("[EOS]")
MAX_SEQ_LEN = 160

class TicketRequest(BaseModel):
    message: str

def generate_ticket_json(message: str, max_new_tokens: int = 110) -> str:
    msg_ids = tokenizer.encode(message.strip()).ids[:48]
    ids = [BOS_ID, MSG_ID] + msg_ids + [JSON_ID]
    prompt_len = len(ids)

    for _ in range(max_new_tokens):
        if len(ids) >= MAX_SEQ_LEN:
            break
        inp = np.array([ids], dtype=np.int64)
        logits = session.run(None, {"input_ids": inp})[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        if next_token == EOS_ID:
            break
        ids.append(next_token)

    return tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()

def parse_json_safe(raw: str) -> tuple[dict, bool]:
    try:
        return json.loads(raw), True
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), False
            except json.JSONDecodeError:
                pass
    return {
        "category": "GENERAL",
        "urgency": "Medium",
        "intent": "general_support",
        "suggested_reply": "Thank you for contacting support; an agent will review your message shortly.",
    }, False

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "model": "TicketGPT-2-ONNX"}

@app.post("/api/triage")
def triage(req: TicketRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    t0 = time.perf_counter()
    raw_output = generate_ticket_json(req.message)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    ticket, strict_valid = parse_json_safe(raw_output)

    return {
        "ticket": ticket,
        "strict_valid_json": strict_valid,
        "latency_ms": latency_ms,
        "raw_output": raw_output,
    }