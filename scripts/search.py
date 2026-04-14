from __future__ import annotations

import argparse

from src.pipeline import VideoRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Поиск по локальному Video-RAG индексу")
    parser.add_argument("query", help="Пользовательский запрос")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    pipeline = VideoRAGPipeline(args.config)
    try:
        decomposition, candidates = pipeline.search(args.query)
    finally:
        pipeline.close()

    print("Query decomposition:")
    print(f"  asr_query: {decomposition.asr_query}")
    print(f"  det_queries: {decomposition.det_queries}")
    print(f"  det_mode: {decomposition.det_mode}")
    print()

    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}] {candidate.video_file}")
        print(f"  span: {candidate.start:.2f} - {candidate.end:.2f}")
        print(f"  score: {candidate.score:.4f}")
        print(candidate.combined_text())
        print()


if __name__ == "__main__":
    main()

