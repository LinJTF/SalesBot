from openai import OpenAI

from config.config import settings


def generate_openai_embeddings(
    texts: list[str], model: str = "text-embedding-3-small"
) -> list[list[float]]:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.embeddings.create(input=texts, model=model)
    return [embedding_data.embedding for embedding_data in response.data]
