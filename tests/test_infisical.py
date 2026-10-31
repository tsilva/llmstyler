"""The deprecated launcher must never retrieve or inject manager secrets."""

import importlib.util
from pathlib import Path

import pytest

from llmstyler.agent import AgentModeError


def test_deprecated_path_is_a_keyless_compatibility_wrapper(capsys):
    path = Path(__file__).parents[1] / "scripts/infisical.py"
    spec = importlib.util.spec_from_file_location("legacy_launcher", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    assert not hasattr(launcher, "read_key")
    assert not hasattr(launcher, "command")
    launcher.main(["check"])
    assert "no service or credential request" in capsys.readouterr().out
    with pytest.raises(AgentModeError):
        launcher.main(["add"])
