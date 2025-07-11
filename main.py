from fastapi import FastAPI, Request
from pydantic import BaseModel
from agents.sales_agent import SalesAgentWorkflow
from llama_index.llms.openai import OpenAI
from tools import query_products, check_user
import json
import re
from fastapi import HTTPException
from llama_index.core.workflow import Context
from config.config import settings

app = FastAPI(title="GenAI Sales Assistant")

# agent = get_sales_agent(tools=[query_products, check_user])

with open(settings.SALER_AGENT_PROMPT_PATH, encoding="utf-8") as f:
    system_prompt = f.read()

agent = SalesAgentWorkflow(
    tools=[query_products, check_user],
    system_prompt=system_prompt,
    timeout=120,
    verbose=True
)

class MessageRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    resposta: str
    cupom_emitido: bool
    intenção: str
    cpf: str
    descricao_escolhida: str
    nome_cupom: str

ctx = Context(agent)


def extract_json(content: str):
    # 1. Primeiro tenta encontrar bloco ```json ... ```
    match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    
    # 2. Se falhar, tenta encontrar o primeiro { ... } como JSON puro
    match = re.search(r"{\s*\"resposta\".*?}", content, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    
    raise HTTPException(400, f"Formato de resposta inesperado da LLM:\n\n{content}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: MessageRequest):
    try:
        handler = await agent.run(input=req.message, ctx=ctx)
        content = handler["response"].message.content
        print(f"Resposta da LLM: {content}")
        
        data = extract_json(content)
        return ChatResponse(**data)

    except Exception as e:
        raise HTTPException(500, f"Erro no agente: {str(e)}")