from __future__ import annotations

import argparse
from pathlib import Path

from .agent import SemanticSQLService
from .providers import DemoProvider, OllamaProvider


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _service(provider_name: str, model: str) -> SemanticSQLService:
    root = _root()
    provider = DemoProvider() if provider_name == "demo" else OllamaProvider(model=model)
    return SemanticSQLService(root / "demo" / "demo.db", provider=provider)


def _print_result(result) -> None:
    print("SQL")
    print(result.candidate.sql if result.candidate else "-")
    print("VERIFY")
    print("PASS" if result.verification and result.verification.ok else result.message)
    print("\nRESULT")
    if result.status == "ok":
        if result.columns:
            print(" | ".join(result.columns))
        for row in result.rows:
            print(" | ".join(str(value) for value in row))
    else:
        print(f"{result.status.upper()}: {result.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="semantic-sql", description="Semantic Text-to-SQL agent demo")
    parser.add_argument("--provider", choices=("demo", "ollama"), default="demo")
    parser.add_argument("--model", default="qwen3.5:4b")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run the showcase Czech Republic query")
    ask = sub.add_parser("ask", help="ask a question against demo.db")
    ask.add_argument("question")
    args = parser.parse_args(argv)

    question = "Show total sales from Czech Republic in 2024" if args.command == "demo" else args.question
    result = _service(args.provider, args.model).ask(question)
    _print_result(result)
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
