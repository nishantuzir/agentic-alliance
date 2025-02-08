import asyncio

from client import AgentClient
from core import settings
from schema import ChatMessage
from langfuse.callback import CallbackHandler

async def amain() -> None:
    #### ASYNC ####
    client = AgentClient(settings.BASE_URL)

    print("Agent info:")
    print(client.info)
    print("Chat example:")
    response = await client.ainvoke("Hi, what can you do?", model="azure-gpt-4o-mini")
    response.pretty_print()

    print("\nStream example:")
    async for message in client.astream("Share a quick fun fact?"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"ERROR: Unknown type - {type(message)}")


def main() -> None:
    #### SYNC ####
    client = AgentClient(settings.BASE_URL)

    print("Agent info:")
    print(client.info)
    print("Chat example:")
    response = client.invoke("Hi, what can you do?", model="azure-gpt-4o-mini")
    response.pretty_print()

    print("\nStream example:")
    for message in client.stream("Share a quick fun fact?"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"ERROR: Unknown type - {type(message)}")


if __name__ == "__main__":
    print("Running in sync mode")
    main()
    print("\n\n\n\n\n")
    print("Running in async mode")
    asyncio.run(amain())
