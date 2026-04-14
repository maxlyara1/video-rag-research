from __future__ import annotations

import argparse

from src.pipeline import VideoRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить индексы Video-RAG по модальностям")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--recreate", action="store_true", help="Пересоздать коллекции Qdrant")
    parser.add_argument("--force", action="store_true", help="Пересчитать артефакты модулей")
    parser.add_argument("--prepare", action="store_true", help="Сначала подготовить видео из lera_materials")
    args = parser.parse_args()

    pipeline = VideoRAGPipeline(args.config)
    try:
        if args.prepare:
            pipeline.prepare_dataset(force=args.force)
        stats = pipeline.build_indexes(recreate=args.recreate, force=args.force)
    finally:
        pipeline.close()

    print("Индексация завершена.")
    for modality, count in stats.items():
        print(f"{modality}: {count}")


if __name__ == "__main__":
    main()

