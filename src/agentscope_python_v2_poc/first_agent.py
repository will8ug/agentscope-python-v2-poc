"""A first AgentScope agent - the quickstart example.

Run (needs ``DASHSCOPE_API_KEY`` in the environment)::

    uv run --env-file .env agentscope-python-v2-poc
    uv run --env-file .env python -m agentscope_python_v2_poc.first_agent

For tool calls, change the prompt to something like:

    List the Python files in the current directory.
"""

import asyncio
import os
import sys

from agentscope.agent import Agent
from agentscope.credential import DashScopeCredential
from agentscope.event import EventType
from agentscope.message import UserMsg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Bash, Edit, Read, Toolkit, Write, Grep, Glob


async def run_awaiting(agent: Agent, user_msg: UserMsg) -> None:
    """Option 1: await the final assistant message.

    ``reply()`` consumes the whole event stream internally and returns
    the final ``Msg``. Its ``.content`` is a list of content blocks,
    so the text is most easily read via ``get_text_content()``.
    """
    reply_msg = await agent.reply(user_msg)
    print(f"[{reply_msg.name}] {reply_msg.get_text_content()}")


async def run_streaming(agent: Agent, user_msg: UserMsg) -> None:
    """Option 2: stream incremental events as the agent reasons and acts.

    Each ``match`` branch handles one event kind; field names are taken
    from the ``AgentEvent`` subclasses in ``agentscope.event``.
    """
    print(f"[user] {user_msg.get_text_content()}\n")

    async for event in agent.reply_stream(user_msg):
        match event.type:
            # --- the visible answer text -------------------------------
            case EventType.TEXT_BLOCK_DELTA:
                print(event.delta, end="", flush=True)
            case EventType.TEXT_BLOCK_END:
                print()  # terminate the streamed paragraph

            # --- reasoning ("thinking") content, e.g. qwen3 models ------
            case EventType.THINKING_BLOCK_START:
                # Start events only carry a `block_id` — the content
                # itself arrives via the DELTA events below.
                print("[thinking] ", end="", flush=True)
            case EventType.THINKING_BLOCK_DELTA:
                print(event.delta, end="", flush=True)
            case EventType.THINKING_BLOCK_END:
                print()

            # --- tool use ------------------------------------------------
            case EventType.TOOL_CALL_START:
                print(
                    f"\n[tool call] {event.tool_call_name} "
                    f"(id={event.tool_call_id[:8]})",
                    flush=True,
                )
            case EventType.TOOL_RESULT_END:
                # Unlike TOOL_RESULT_START, this event carries no
                # `tool_call_name`; `state` is e.g. "success"/"error".
                print(
                    f"[tool result] {event.tool_call_id[:8]} -> "
                    f"{event.state}",
                    flush=True,
                )

            # --- lifecycle & usage ----------------------------------------
            case EventType.MODEL_CALL_END:
                print(
                    f"[usage] in={event.input_tokens} "
                    f"out={event.output_tokens} tokens",
                    flush=True,
                )
            case EventType.REPLY_END:
                print(f"[reply] finished_reason={event.finished_reason}")

            # --- chatty deltas not worth echoing --------------------------
            case (
                EventType.TOOL_CALL_DELTA
                | EventType.TOOL_RESULT_TEXT_DELTA
                | EventType.TOOL_RESULT_DATA_DELTA
            ):
                pass

            # --- anything else --------------------------------------------
            case _:
                # e.g. REPLY_START, MODEL_CALL_START, TEXT_BLOCK_START,
                # DATA_BLOCK_*, HINT_BLOCK, TOOL_CALL_END, ...
                print(f"[event] {event.type}", flush=True)


async def main() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        sys.exit(
            "DASHSCOPE_API_KEY is not set — export it, or put it in a "
            ".env file and run:\n"
            "  uv run --env-file .env python -m "
            "agentscope_python_v2_poc.first_agent",
        )

    agent = Agent(
        name="Friday",
        system_prompt="You are a helpful assistant named Friday.",
        model=DashScopeChatModel(
            credential=DashScopeCredential(api_key=api_key),
            model="qwen3.8-flash",
        ),
        toolkit=Toolkit(tools=[Bash(), Read(), Write(), Edit(), Grep(), Glob()]),
    )

    # user_prompt = "Hello, who are you?"
    user_prompt = "List the Python files in the current directory."
    user_msg = UserMsg(name="user", content=user_prompt)

    # Option 1: await the final assistant message.
    # await run_awaiting(agent, user_msg)

    # Option 2: stream incremental events (text deltas, tool calls, ...).
    await run_streaming(agent, user_msg)


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
