from __future__ import annotations

import argparse

from src.pipeline import VideoRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка видео из lera_materials")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true", help="Пересобрать каталог data/videos с нуля")
    args = parser.parse_args()

    pipeline = VideoRAGPipeline(args.config)
    try:
        videos = pipeline.prepare_dataset(force=args.force)
    finally:
        pipeline.close()

    print(f"Подготовлено видео: {len(videos)}")
    for path in videos:
        print(path)


if __name__ == "__main__":
    main()

