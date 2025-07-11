from typing import Any, List

from config.config import settings
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from llama_index.core.tools import ToolSelection
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
)

from llama_index.llms.openai import OpenAI


class InputEvent(Event):
    input: list[ChatMessage]


class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class SalesAgentWorkflow(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        tools: List[BaseTool] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm or OpenAI(model=settings.LLM_MODEL_NAME,system_prompt=system_prompt)
        assert self.llm.metadata.is_function_calling_model

    @step
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        await ctx.store.set("sources", [])
        memory = await ctx.store.get("memory", default=None)

        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)
            if self.llm.system_prompt:
                memory.put(ChatMessage(role="system", content=self.llm.system_prompt))

        user_msg = ChatMessage(role="user", content=ev.input)
        memory.put(user_msg)
        await ctx.store.set("memory", memory)

        return InputEvent(input=memory.get())

    @step
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> StopEvent | ToolCallEvent:
        chat_history = ev.input
        response_stream = await self.llm.astream_chat_with_tools(self.tools, chat_history=chat_history)

        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        memory = await ctx.store.get("memory")
        memory.put(response.message)
        await ctx.store.set("memory", memory)

        tool_calls = self.llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
        if not tool_calls:
            sources = await ctx.store.get("sources", default=[])
            return StopEvent(result={"response": response, "sources": sources})

        return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> StopEvent:
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        sources = await ctx.store.get("sources", default=[])
        memory = await ctx.store.get("memory")

        # 1️⃣ Executa as ferramentas
        tool_msgs = []
        for tool_call in ev.tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            kwargs = {"tool_call_id": tool_call.tool_id, "name": tool.metadata.get_name()}

            if not tool:
                tool_msgs.append(ChatMessage(role="tool", content=f"Tool {tool_call.tool_name} does not exist", additional_kwargs=kwargs))
                continue

            try:
                print(f'\n Executando ferramenta: {tool.metadata.get_name()} com args: {tool_call.tool_kwargs}')
                tool_output = tool(**tool_call.tool_kwargs)
                
                # Fix: tool_output é uma string, não um objeto com .content
                if hasattr(tool_output, 'content'):
                    content = tool_output.content
                else:
                    content = str(tool_output)
                
                sources.append({"content": content, "tool": tool_call.tool_name})
                tool_msgs.append(ChatMessage(role="tool", content=content, additional_kwargs=kwargs))
            except Exception as e:
                error_msg = f"Erro na ferramenta {tool_call.tool_name}: {e}"
                tool_msgs.append(ChatMessage(role="tool", content=error_msg, additional_kwargs=kwargs))

        for msg in tool_msgs:
            memory.put(msg)

        await ctx.store.set("memory", memory)
        await ctx.store.set("sources", sources)

        chat_history = memory.get()

        print(f'\n AQUI ESTA O HISTÓRICO DE CHAT: {chat_history}')

        response_stream = await self.llm.astream_chat_with_tools(self.tools, chat_history=chat_history)

        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        memory.put(response.message)
        await ctx.store.set("memory", memory)

        print(f'\n AQUI ESTA A RESPOSTA: {response.message.content}')

        return StopEvent(result={"response": response, "sources": sources})
    
    
