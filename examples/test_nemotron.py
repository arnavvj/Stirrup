"""Quick test script using NVIDIA Nemotron 3 Nano 30B (free) via OpenRouter.

Setup:
    1. Get a free OpenRouter API key at https://openrouter.ai/keys
    2. Copy .env.example to .env in the repo root:
           cp .env.example .env
    3. Set your key in .env:
           OPENROUTER_API_KEY=sk-or-v1-...
    4. Run with UTF-8 encoding (required on Windows):
           PYTHONUTF8=1 python examples/test_nemotron.py

Expected output:
    - A Rich-formatted agent session log in the terminal showing each turn:
        * Turn 1: agent writes fib_test.py via code_exec
        * Turn 2: agent runs the file and gets "Fibonacci(10) = 55"
        * Turn 3+: agent calls finish() to complete the task
    - A summary panel showing tool usage, token counts, and model speed (~150-250 tok/s)
    - The result printed at the end:
        --- Result ---
        Task completed: provided a Python function that returns the nth Fibonacci
        number and includes a test that verifies the result for n=10 (output 55).
        --- Token Usage ---
        [TokenUsage(input=..., answer=..., reasoning=...), ...]
    - Output file saved to: output/nemotron_test/fib_test.py
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient


async def main() -> None:
    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="nvidia/nemotron-3-nano-30b-a3b:free",
    )

    agent = Agent(client=client, name="agent", max_turns=10)

    async with agent.session(output_dir="output/nemotron_test") as session:
        finish_params, _history, metadata = await session.run(
            "Write a Python function that returns the nth Fibonacci number, then test it for n=10."
        )

    print("\n--- Result ---")
    print(finish_params.reason if finish_params else "Agent did not finish (max turns reached)")
    print("\n--- Token Usage ---")
    print(metadata.get("token_usage"))


if __name__ == "__main__":
    asyncio.run(main())
