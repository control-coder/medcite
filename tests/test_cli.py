from click.testing import CliRunner

from medidiag.cli import main


def test_demo_help_shows_non_reserved_default_port() -> None:
    result = CliRunner().invoke(main, ["demo", "--help"])

    assert result.exit_code == 0
    assert "[default: 8400; 1<=x<=65535]" in result.output


def test_demo_retrieval_mock_has_accurate_runtime(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    import uvicorn

    from medidiag import cli
    observed = []
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(
        database_url="sqlite:///" + (tmp_path / "demo.db").as_posix(), medidiag_max_review_rounds=3))
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(threading.Thread, "join", lambda self, **kwargs: None)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: observed.append(app.state.demo_runtime))
    result = CliRunner().invoke(main, ["demo", "--provider", "retrieval_mock"])
    assert result.exit_code == 0, result.output
    assert "模拟生成" in observed[0]["label"]
    assert "未运行 NLI" in observed[0]["detail"]
