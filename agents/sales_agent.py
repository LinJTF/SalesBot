from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from config.config import settings
from langfuse_logger import langfuse


def get_sales_agent(tools=None):
    with open(settings.SALER_AGENT_PROMPT_PATH, encoding="utf-8") as f:
        system_prompt = f.read()

    llm = OpenAI(model=settings.LLM_MODEL_NAME, temperature=settings.TEMPERATURE)

    return OpenAIAgent.from_tools(
        tools=tools or [],
        llm=llm,
        system_prompt=system_prompt,
        verbose=True
    )
