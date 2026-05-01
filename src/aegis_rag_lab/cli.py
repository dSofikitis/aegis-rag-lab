import argparse
from pathlib import Path

from aegis_rag_lab.config import get_settings
from aegis_rag_lab.logging import configure_logging
from aegis_rag_lab.rag.ingestion import load_documents_from_path
from aegis_rag_lab.rag.service import RagService
from aegis_rag_lab.eval.harness import run_eval


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Aegis RAG Lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from disk")
    ingest_parser.add_argument("--path", required=True, help="Path to docs")
    ingest_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for documents",
    )
    ingest_parser.add_argument(
        "--extensions",
        default=".md,.txt,.jsonl",
        help="Comma-separated list of extensions",
    )

    eval_parser = subparsers.add_parser("eval", help="Run evaluation harness")
    eval_parser.add_argument(
        "--dataset",
        default="data/eval/sample_eval.jsonl",
        help="Path to evaluation dataset",
    )
    eval_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM calls and score retrieval only",
    )

    args = parser.parse_args()
    settings = get_settings()
    service = RagService(settings)
    service.ensure_ready()

    if args.command == "ingest":
        extensions = [ext.strip() for ext in args.extensions.split(",") if ext.strip()]
        documents = load_documents_from_path(
            Path(args.path),
            recursive=args.recursive,
            extensions=extensions,
        )
        result = service.ingest_documents(documents)
        print(f"Ingested {result['documents']} documents ({result['chunks']} chunks).")
        return

    if args.command == "eval":
        report = run_eval(
            service=service,
            dataset_path=Path(args.dataset),
            use_llm=not args.no_llm,
        )
        print(report)


if __name__ == "__main__":
    main()
