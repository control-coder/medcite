"""CLI 入口占位。

阶段 0 仅提供 ``medidiag`` 命令骨架，子命令在后续阶段补充。
"""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """MediDiag-Agent EvidenceFlow CLI."""


@main.command()
def version() -> None:
    """显示版本号。"""
    from medidiag import __version__

    click.echo(__version__)


@main.command()
def errors() -> None:
    """列出所有已注册的错误码。"""
    from medidiag.errors import all_error_codes

    for code, spec in all_error_codes().items():
        click.echo(
            f"{code:35s} [{spec.category.value}] "
            f"retryable={spec.retryable!s:5s} "
            f"escalate={spec.requires_human_escalation!s:5s} "
            f"alert={spec.alert!s:5s} "
            f"http={spec.http_status}"
        )


if __name__ == "__main__":
    main()
