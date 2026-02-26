"""Tests for agent core functionality."""

import asyncio
import logging
from pathlib import Path

import pytest
from pydantic import BaseModel

from stirrup.constants import FINISH_TOOL_NAME
from stirrup.core.agent import Agent, _PARENT_DEPTH
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.utils.logging import AgentLoggerBase
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    LLMClient,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from stirrup.tools.finish import SIMPLE_FINISH_TOOL, FinishParams


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = responses
        self.call_count = 0

    @property
    def model_slug(self) -> str:
        return "mock-model"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:  # noqa: ARG002
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


class NullAgentLogger(AgentLoggerBase):
    """No-op logger for tests that need nested agent sessions without Rich display conflicts."""

    name: str = ""
    model: str | None = None
    max_turns: int | None = None
    depth: int = 0
    finish_params = None
    run_metadata = None
    output_dir: str | None = None

    def __enter__(self) -> "NullAgentLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        pass

    def on_step(self, step: int, tool_calls: int = 0, input_tokens: int = 0, output_tokens: int = 0) -> None:  # noqa: ARG002
        pass

    def assistant_message(self, turn: int, max_turns: int, assistant_message: AssistantMessage) -> None:  # noqa: ARG002
        pass

    def user_message(self, user_message: UserMessage) -> None:  # noqa: ARG002
        pass

    def task_message(self, task) -> None:  # noqa: ARG002, ANN001
        pass

    def tool_result(self, tool_message) -> None:  # noqa: ARG002, ANN001
        pass

    def context_summarization_start(self, pct_used: float, cutoff: float) -> None:  # noqa: ARG002
        pass

    def context_summarization_complete(self, summary: str, bridge: str) -> None:  # noqa: ARG002
        pass

    def debug(self, message: str, *args: object) -> None:  # noqa: ARG002
        pass

    def info(self, message: str, *args: object) -> None:  # noqa: ARG002
        pass

    def warning(self, message: str, *args: object) -> None:  # noqa: ARG002
        pass

    def error(self, message: str, *args: object) -> None:  # noqa: ARG002
        pass


async def test_agent_basic_finish() -> None:
    """Test agent completes successfully when finish tool is called."""
    # Create mock responses
    responses = [
        AssistantMessage(
            content="I'll finish now",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Task completed successfully", "paths": []}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
            request_start_time=100.0,
            request_end_time=100.4,
        )
    ]

    # Create agent with mock client
    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    # Run agent with session context
    async with agent.session() as session:
        finish_params, message_history, run_metadata = await session.run(
            [
                SystemMessage(content="Test system message"),
                UserMessage(content="Test task"),
            ]
        )

    # Assertions
    assert finish_params is not None
    assert isinstance(finish_params, FinishParams)
    assert finish_params.reason == "Task completed successfully"
    assert isinstance(run_metadata, dict)
    # Agent's own token usage metadata should be present
    assert "token_usage" in run_metadata
    assert len(message_history) == 1  # One turn
    assert client.call_count == 1


async def test_agent_max_turns() -> None:
    """Test agent stops after max_turns is reached."""
    # Create mock responses (never calls finish)
    responses = [
        AssistantMessage(
            content=f"Turn {i}",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        )
        for i in range(5)
    ]

    # Create agent with mock client
    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=3,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    # Run agent with session context
    async with agent.session() as session:
        finish_params, _message_history, run_metadata = await session.run(
            [
                SystemMessage(content="Test system message"),
                UserMessage(content="Test task"),
            ]
        )

    # Assertions
    assert finish_params is None  # Did not finish
    assert client.call_count == 3  # Ran max_turns times
    assert isinstance(run_metadata, dict)
    # Agent's own token usage metadata should be present
    assert "token_usage" in run_metadata


async def test_agent_tool_execution() -> None:
    """Test agent executes custom tools correctly."""

    class EchoParams(BaseModel):
        message: str

    def echo_executor(params: EchoParams) -> ToolResult:
        return ToolResult(content=f"Echo: {params.message}")

    echo_tool = Tool[EchoParams, None](
        name="echo",
        description="Echo a message",
        parameters=EchoParams,
        executor=echo_executor,  # ty: ignore[invalid-argument-type]
    )

    # Create mock responses
    responses = [
        # First turn: call echo tool
        AssistantMessage(
            content="I'll echo your message",
            tool_calls=[
                ToolCall(
                    name="echo",
                    arguments='{"message": "Hello"}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Second turn: finish
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Echoed successfully", "paths": []}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    # Create agent with mock client
    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[echo_tool],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    # Run agent with session context
    async with agent.session() as session:
        finish_params, message_history, run_metadata = await session.run(
            [
                SystemMessage(content="Test system message"),
                UserMessage(content="Echo 'Hello'"),
            ]
        )

    # Assertions
    assert finish_params is not None
    assert finish_params.reason == "Echoed successfully"
    assert client.call_count == 2
    # Check that run metadata tracks called tools
    assert "echo" in run_metadata
    assert isinstance(run_metadata["echo"], list)
    # Agent's own token usage metadata should be present
    assert "token_usage" in run_metadata
    # Check that tool was executed
    messages = message_history[0]
    tool_messages: list[ToolMessage] = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2  # Echo tool + finish tool
    # Find the echo tool message
    echo_messages = [m for m in tool_messages if m.name == "echo"]
    assert len(echo_messages) == 1
    assert "Echo: Hello" in echo_messages[0].content


async def test_agent_invalid_tool_call() -> None:
    """Test agent handles invalid tool calls gracefully."""
    # Create mock responses
    responses = [
        # Call non-existent tool
        AssistantMessage(
            content="I'll call a tool",
            tool_calls=[
                ToolCall(
                    name="nonexistent_tool",
                    arguments='{"param": "value"}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Then finish
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Handled error", "paths": []}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    # Create agent with mock client
    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    # Run agent with session context
    async with agent.session() as session:
        finish_params, message_history, run_metadata = await session.run(
            [
                SystemMessage(content="Test system message"),
                UserMessage(content="Test task"),
            ]
        )

    # Assertions
    assert finish_params is not None
    assert finish_params.reason == "Handled error"
    # Nonexistent tool should still be tracked (with empty metadata list)
    assert "nonexistent_tool" in run_metadata
    # Agent's own token usage metadata should be present
    assert "token_usage" in run_metadata
    # Check that tool error message was returned
    messages = message_history[0]
    tool_messages: list[ToolMessage] = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2  # Error message + finish tool
    # Find the error tool message
    error_messages = [m for m in tool_messages if m.name == "nonexistent_tool"]
    assert len(error_messages) == 1
    assert "not a valid tool" in error_messages[0].content


async def test_agent_finish_tool_validation() -> None:
    """Test agent only terminates on valid finish tool calls."""
    from stirrup.core.models import ToolUseCountMetadata

    class CustomFinishParams(BaseModel):
        reason: str
        status: str

    # Custom finish tool that validates status before allowing termination
    def custom_finish_executor(params: CustomFinishParams) -> ToolResult[ToolUseCountMetadata]:
        is_valid = params.status == "complete"
        return ToolResult(
            content=params.reason,
            success=is_valid,
            metadata=ToolUseCountMetadata(),
        )

    custom_finish_tool = Tool[CustomFinishParams, ToolUseCountMetadata](
        name=FINISH_TOOL_NAME,
        description="Finish with status validation",
        parameters=CustomFinishParams,
        executor=custom_finish_executor,
    )

    # Create mock responses
    responses = [
        # First: invalid finish (status != "complete")
        AssistantMessage(
            content="Trying to finish",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Not ready", "status": "pending"}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Second: valid finish (status == "complete")
        AssistantMessage(
            content="Now finishing",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Task done", "status": "complete"}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=custom_finish_tool,
    )

    async with agent.session() as session:
        finish_params, _, _ = await session.run([UserMessage(content="Test task")])

    # Agent should have taken 2 turns (invalid finish + valid finish)
    assert client.call_count == 2
    assert finish_params is not None
    assert finish_params.reason == "Task done"
    assert finish_params.status == "complete"


async def test_finish_tool_validates_file_paths() -> None:
    """Test that SIMPLE_FINISH_TOOL rejects non-existent file paths."""
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

    # Create mock responses
    responses = [
        # First: finish with non-existent file path
        AssistantMessage(
            content="Finishing with fake file",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Done", "paths": ["nonexistent.txt"]}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Second: finish with empty paths (should succeed)
        AssistantMessage(
            content="Finishing properly",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Actually done", "paths": []}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[LocalCodeExecToolProvider()],
    )

    async with agent.session() as session:
        finish_params, history, _ = await session.run([UserMessage(content="Test task")])

    # Agent should have taken 2 turns (failed finish + successful finish)
    assert client.call_count == 2
    assert finish_params is not None
    assert finish_params.reason == "Actually done"

    # First finish should have failed with error about missing file
    tool_messages = [msg for group in history for msg in group if isinstance(msg, ToolMessage)]
    assert any("nonexistent.txt" in str(msg.content) and not msg.success for msg in tool_messages)


async def test_no_successive_assistant_messages() -> None:
    """Test agent adds continue message to avoid successive assistant messages."""
    responses = [
        # First: assistant message without tool calls
        AssistantMessage(
            content="Let me think about this",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Second: finish after continue
        AssistantMessage(
            content="Now I'll finish",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Task completed", "paths": []}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=30,  # Use default max_turns so warning threshold won't be hit
        turns_remaining_warning_threshold=5,  # Only warn in last 5 turns
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with agent.session() as session:
        finish_params, message_history, _ = await session.run([UserMessage(content="Test task")])

    # Verify finish params
    assert finish_params is not None
    assert finish_params.reason == "Task completed"
    assert client.call_count == 2

    # Verify "Please continue the task" message was added after first assistant message
    messages = message_history[0]
    continue_messages = [m for m in messages if isinstance(m, UserMessage) and m.content == "Please continue the task"]
    assert len(continue_messages) == 1


async def test_allow_successive_assistant_messages() -> None:
    """Test agent allows successive assistant messages when flag is enabled."""
    responses = [
        # First: assistant message without tool calls
        AssistantMessage(
            content="Let me think about this",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Second: another assistant message without continue prompt
        AssistantMessage(
            content="Now I'll finish",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Task completed", "paths": []}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=30,
        turns_remaining_warning_threshold=5,
        block_successive_assistant_messages=False,  # Disable blocking
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with agent.session() as session:
        finish_params, message_history, _ = await session.run([UserMessage(content="Test task")])

    # Verify finish params
    assert finish_params is not None
    assert finish_params.reason == "Task completed"
    assert client.call_count == 2

    # Verify NO "Please continue the task" message was added
    messages = message_history[0]
    continue_messages = [m for m in messages if isinstance(m, UserMessage) and m.content == "Please continue the task"]
    assert len(continue_messages) == 0


async def test_agent_duplicate_tool_names_raises() -> None:
    """Agent raises ValueError at session entry if two tools share a name."""
    dup_tool = Tool[dict, None](
        name="my_tool",
        description="first",
        parameters=dict,
        executor=lambda _: ToolResult(content="ok"),
    )
    dup_tool2 = Tool[dict, None](
        name="my_tool",  # same name — should trigger validation error
        description="second",
        parameters=dict,
        executor=lambda _: ToolResult(content="ok"),
    )

    agent = Agent(
        client=MockLLMClient(responses=[]),
        name="test_agent",
        tools=[dup_tool, dup_tool2],
    )

    with pytest.raises(ValueError, match="duplicate tool names"):
        async with agent.session():
            pass


async def test_agent_unique_tool_names_ok() -> None:
    """Agent with all unique tool names enters session without error."""
    client = MockLLMClient(
        responses=[
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name=FINISH_TOOL_NAME,
                        arguments='{"reason": "done", "paths": []}',
                        tool_call_id="call_1",
                    )
                ],
                token_usage=TokenUsage(input=10, answer=5),
            )
        ]
    )
    tool_a = Tool[dict, None](
        name="tool_a",
        description="a",
        parameters=dict,
        executor=lambda _: ToolResult(content="ok"),
    )
    tool_b = Tool[dict, None](
        name="tool_b",
        description="b",
        parameters=dict,
        executor=lambda _: ToolResult(content="ok"),
    )

    agent = Agent(client=client, name="test_agent", tools=[tool_a, tool_b])
    async with agent.session() as session:
        finish_params, _, _ = await session.run("go")

    assert finish_params is not None
    assert finish_params.reason == "done"


async def test_session_output_dir_gets_session_subdir(tmp_path: Path) -> None:
    """A root session with output_dir resolves to a session-<id> subdirectory."""
    client = MockLLMClient(
        responses=[
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name=FINISH_TOOL_NAME,
                        arguments='{"reason": "done", "paths": []}',
                        tool_call_id="call_1",
                    )
                ],
                token_usage=TokenUsage(input=10, answer=5),
            )
        ]
    )
    agent = Agent(client=client, name="test_agent", tools=[])

    async with agent.session(output_dir=tmp_path) as session:
        await session.run("go")

    # After session exit, _logger.output_dir reflects the actual per-session path used
    actual_output_dir = agent._logger.output_dir  # noqa: SLF001
    assert actual_output_dir is not None
    assert "session-" in actual_output_dir
    # Must be a direct subdirectory of the given output_dir
    assert Path(actual_output_dir).parent == tmp_path


async def test_concurrent_sessions_get_distinct_subdirs(tmp_path: Path) -> None:
    """Two concurrent sessions with the same output_dir receive distinct session-<id> subdirs."""

    def make_client() -> MockLLMClient:
        return MockLLMClient(
            responses=[
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name=FINISH_TOOL_NAME,
                            arguments='{"reason": "done", "paths": []}',
                            tool_call_id="call_1",
                        )
                    ],
                    token_usage=TokenUsage(input=10, answer=5),
                )
            ]
        )

    agent_a = Agent(client=make_client(), name="agent_a", tools=[])
    agent_b = Agent(client=make_client(), name="agent_b", tools=[])

    actual_dirs: list[str] = []

    async def run_session(agent: Agent) -> None:
        async with agent.session(output_dir=tmp_path) as session:
            await session.run("go")
        actual_dirs.append(agent._logger.output_dir)  # noqa: SLF001

    await asyncio.gather(run_session(agent_a), run_session(agent_b))

    assert len(actual_dirs) == 2
    assert all("session-" in d for d in actual_dirs)
    assert actual_dirs[0] != actual_dirs[1], "Concurrent sessions must get distinct subdirectories"


# ---------------------------------------------------------------------------
# Unit tests for _resolve_input_files
# ---------------------------------------------------------------------------


def test_resolve_input_files_glob_returns_matches(tmp_path: Path) -> None:
    """_resolve_input_files expands a glob pattern to the matching file paths."""
    (tmp_path / "a.csv").write_text("a")
    (tmp_path / "b.csv").write_text("b")
    (tmp_path / "other.txt").write_text("x")

    agent = Agent(client=MockLLMClient(responses=[]), name="test_agent", tools=[])
    resolved = agent._resolve_input_files(str(tmp_path / "*.csv"))  # noqa: SLF001

    assert sorted(p.name for p in resolved) == ["a.csv", "b.csv"]


def test_resolve_input_files_empty_glob_raises(tmp_path: Path) -> None:
    """_resolve_input_files raises ValueError when a glob pattern matches no files."""
    agent = Agent(client=MockLLMClient(responses=[]), name="test_agent", tools=[])

    with pytest.raises(ValueError, match="matched no files"):
        agent._resolve_input_files(str(tmp_path / "*.csv"))  # noqa: SLF001


def test_resolve_input_files_non_glob_passthrough(tmp_path: Path) -> None:
    """_resolve_input_files returns non-glob paths as-is without checking existence."""
    nonexistent = tmp_path / "no_such_file.txt"

    agent = Agent(client=MockLLMClient(responses=[]), name="test_agent", tools=[])
    resolved = agent._resolve_input_files(nonexistent)  # noqa: SLF001

    assert resolved == [nonexistent]


def test_resolve_input_files_mixed_list(tmp_path: Path) -> None:
    """_resolve_input_files handles a list mixing glob patterns and plain paths."""
    (tmp_path / "a.csv").write_text("a")
    plain = tmp_path / "plain.txt"
    plain.write_text("p")

    agent = Agent(client=MockLLMClient(responses=[]), name="test_agent", tools=[])
    resolved = agent._resolve_input_files([str(tmp_path / "*.csv"), plain])  # noqa: SLF001

    assert sorted(p.name for p in resolved) == ["a.csv", "plain.txt"]


async def test_session_empty_glob_raises_before_run(tmp_path: Path) -> None:
    """session() raises ValueError at entry if an input_files glob matches nothing."""
    agent = Agent(
        client=MockLLMClient(responses=[]),
        name="test_agent",
        tools=[LocalCodeExecToolProvider()],
    )

    with pytest.raises(ValueError, match="matched no files"):
        async with agent.session(
            output_dir=tmp_path,
            input_files=str(tmp_path / "*.csv"),  # no CSV files exist
        ):
            pass  # should never reach here


# ---------------------------------------------------------------------------
# Sub-agent edge case tests
# ---------------------------------------------------------------------------


async def test_subagent_with_code_exec_requires_parent_code_exec() -> None:
    """Parent without code exec raises ValueError at session entry if subagent has one."""
    # Sub-agent HAS a code exec tool
    sub_agent = Agent(
        client=MockLLMClient(responses=[]),
        name="sub_agent",
        tools=[LocalCodeExecToolProvider()],
    )
    # Parent does NOT have a code exec tool
    parent_agent = Agent(
        client=MockLLMClient(responses=[]),
        name="parent_agent",
        tools=[sub_agent.to_tool(description="sub")],
    )

    with pytest.raises(ValueError, match="code execution tool"):
        async with parent_agent.session():
            pass


async def test_deeply_nested_subagent_runs_successfully() -> None:
    """A 3-level agent chain (root→child→grandchild) runs without error."""

    def make_finish_client() -> MockLLMClient:
        return MockLLMClient(
            responses=[
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name=FINISH_TOOL_NAME,
                            arguments='{"reason": "done", "paths": []}',
                            tool_call_id="call_1",
                        )
                    ],
                    token_usage=TokenUsage(input=10, answer=5),
                )
            ]
        )

    grandchild_client = make_finish_client()
    grandchild = Agent(client=grandchild_client, name="grandchild", tools=[], logger=NullAgentLogger())

    child_client = MockLLMClient(
        responses=[
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="grandchild",
                        arguments='{"task": "do something"}',
                        tool_call_id="call_1",
                    )
                ],
                token_usage=TokenUsage(input=10, answer=5),
            ),
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name=FINISH_TOOL_NAME,
                        arguments='{"reason": "done", "paths": []}',
                        tool_call_id="call_2",
                    )
                ],
                token_usage=TokenUsage(input=10, answer=5),
            ),
        ]
    )
    child = Agent(
        client=child_client,
        name="child",
        tools=[grandchild.to_tool(description="grandchild agent")],
        logger=NullAgentLogger(),
    )

    root_client = MockLLMClient(
        responses=[
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="child",
                        arguments='{"task": "do something"}',
                        tool_call_id="call_1",
                    )
                ],
                token_usage=TokenUsage(input=10, answer=5),
            ),
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name=FINISH_TOOL_NAME,
                        arguments='{"reason": "done", "paths": []}',
                        tool_call_id="call_2",
                    )
                ],
                token_usage=TokenUsage(input=10, answer=5),
            ),
        ]
    )
    root = Agent(
        client=root_client,
        name="root",
        tools=[child.to_tool(description="child agent")],
        logger=NullAgentLogger(),
    )

    async with root.session() as session:
        finish_params, _, _ = await session.run("do something")

    assert finish_params is not None
    assert grandchild_client.call_count == 1  # grandchild was actually invoked


async def test_subagent_without_parent_exec_env_warns(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Subagent at depth>0 with exec_env but no parent_exec_env emits WARNING, not exception."""
    # Two-step responses: first create a file, then finish referencing it.
    # The file must exist in the exec env for finish tool validation to pass,
    # which ensures paths is non-empty and the warning code path is reached.
    # NullAgentLogger is required so pytest caplog captures the stdlib logger.warning()
    # call. The default AgentLogger installs a RichHandler that clears caplog's handler.
    sub_agent = Agent(
        client=MockLLMClient(
            responses=[
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="code_exec",
                            arguments='{"cmd": "echo hello > output.txt"}',
                            tool_call_id="call_1",
                        )
                    ],
                    token_usage=TokenUsage(input=10, answer=5),
                ),
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name=FINISH_TOOL_NAME,
                            arguments='{"reason": "done", "paths": ["output.txt"]}',
                            tool_call_id="call_2",
                        )
                    ],
                    token_usage=TokenUsage(input=10, answer=5),
                ),
            ]
        ),
        name="sub_agent",
        tools=[LocalCodeExecToolProvider()],
        logger=NullAgentLogger(),
    )

    # Force depth=1 so __aenter__ treats this as a subagent session.
    # _SESSION_STATE has no parent state → parent_exec_env will be None.
    token = _PARENT_DEPTH.set(1)
    try:
        with caplog.at_level(logging.WARNING, logger="stirrup.core.agent"):
            async with sub_agent.session(output_dir=tmp_path) as session:
                await session.run("go")
    finally:
        _PARENT_DEPTH.reset(token)

    # Should emit the warning about missing parent_exec_env, but NOT raise an exception
    assert any("no parent_exec_env" in r.message for r in caplog.records)
