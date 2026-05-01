from types import SimpleNamespace
from unittest.mock import MagicMock


def test_compress_context_touches_activity_before_blocking_compressor():
    """Compression can be slow; it must reset gateway inactivity watchdog state."""
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.session_id = "sess-1"
    agent.model = "test-model"
    agent.platform = "discord"
    agent.tools = []
    agent._memory_manager = None
    agent._session_db = None
    agent._last_activity_ts = 0.0
    agent._last_activity_desc = "idle"
    agent._last_compression_summary_warning = None
    agent._last_aux_fallback_warning_key = None
    agent._todo_store = SimpleNamespace(format_for_injection=lambda: "")
    agent._invalidate_system_prompt = MagicMock()
    agent._build_system_prompt = MagicMock(return_value="new system")
    agent._cached_system_prompt = None
    agent._emit_warning = MagicMock()
    agent._vprint = MagicMock()

    compressor = MagicMock()
    compressor.compress.return_value = [{"role": "user", "content": "compressed"}]
    compressor._last_summary_error = None
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    agent.context_compressor = compressor

    compressed, new_system = agent._compress_context(
        [{"role": "user", "content": "hello"}],
        "system",
        approx_tokens=123,
    )

    assert compressed == [{"role": "user", "content": "compressed"}]
    assert new_system == "new system"
    assert agent._last_activity_desc == "compressing context"
    assert agent._last_activity_ts > 0
    compressor.compress.assert_called_once()
