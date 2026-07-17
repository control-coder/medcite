from click.testing import CliRunner

from medidiag.cli import main


def test_demo_help_shows_non_reserved_default_port() -> None:
    result = CliRunner().invoke(main, ["demo", "--help"])

    assert result.exit_code == 0
    assert "[default: 8400; 1<=x<=65535]" in result.output
