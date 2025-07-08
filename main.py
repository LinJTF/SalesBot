from fastapi import FastAPI, Request
from pydantic import BaseModel
from agents.sales_agent import get_sales_agent
from tools import query_products, check_user
import json
import re
from fastapi import HTTPException


app = FastAPI(title="GenAI Sales Assistant")

agent = get_sales_agent(tools=[query_products, check_user])

class MessageRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    resposta: str
    cupom_emitido: bool
    intenção: str
    cpf: str
    descricao_escolhida: str
    nome_cupom: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: MessageRequest):
    raw = await agent.achat(req.message)
    match = re.search(r"```json\n(.*?)\n```", str(raw), re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(1))
            return ChatResponse(**data)
        except Exception:
            raise HTTPException(500, "Erro ao interpretar resposta da LLM")
    
    raise HTTPException(400, "Formato de resposta inesperado")