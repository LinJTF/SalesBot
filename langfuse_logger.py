from langfuse import get_client
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
 
LlamaIndexInstrumentor().instrument()

langfuse = get_client()

if langfuse.auth_check():
    print('Langfuse client initialized successfully.')
else:
    print('Failed to initialize Langfuse client. Please check your configuration.')