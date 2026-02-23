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
        * Turns 1-10: agent writes each sorting algorithm file and runs it
        * Final turn: agent calls finish() to complete the task
    - A summary panel showing tool usage, token counts, and model speed (~150-250 tok/s)
    - The result printed at the end:
        --- Result ---
        Task completed: wrote and verified 5 sorting algorithms ...
        --- Token Usage ---
        [TokenUsage(input=..., answer=..., reasoning=...), ...]
    - Output files saved to: output/nemotron_test/ (bubble_sort.py, merge_sort.py, etc.)
    - If interrupted, re-running will resume from the last completed turn (resume=True)
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

    agent = Agent(client=client, name="agent", max_turns=20)

    async with agent.session(output_dir="output/nemotron_test", resume=True) as session:
        finish_params, _history, metadata = await session.run(
            "Write 5 different Python sorting algorithms (bubble, merge, quick, heap, insertion), "
            "each in its own file (bubble_sort.py, merge_sort.py, quick_sort.py, heap_sort.py, "
            "insertion_sort.py), with a test for each algorithm in the same file. "
            "Run each file to verify the tests pass."
        )

    print("\n--- Result ---")
    print(finish_params.reason if finish_params else "Agent did not finish (max turns reached)")
    print("\n--- Token Usage ---")
    print(metadata.get("token_usage"))


if __name__ == "__main__":
    asyncio.run(main())
