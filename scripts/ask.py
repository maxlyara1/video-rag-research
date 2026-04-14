from __future__ import annotations

import argparse

from src.pipeline import VideoRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный Video-RAG: поиск фрагментов и финальный ответ через LVLM")
    parser.add_argument("query", help="Пользовательский запрос")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--show-context", action="store_true", help="Показать найденные фрагменты после ответа")
    args = parser.parse_args()

    pipeline = VideoRAGPipeline(args.config)
    try:
        decomposition, candidates, answer, model_name, key_index = pipeline.answer(args.query)
    finally:
        pipeline.close()

    print(answer)
    print()
    print(f"Gemini model: {model_name or 'unknown'}")
    if key_index is not None:
        print(f"Gemini key index: {key_index}")

    if args.show_context:
        print()
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
