from __future__ import annotations

import sys
from pathlib import Path
from src.pipeline import VideoRAGPipeline

def run_smoke_test() -> None:
    print("=== Запуск Smoke-теста Video-RAG Pipeline ===")
    
    config_path = "configs/config.yaml"
    if not Path(config_path).exists():
        print(f"Ошибка: Конфиг {config_path} не найден!")
        sys.exit(1)
        
    print("Инициализация пайплайна...")
    pipeline = VideoRAGPipeline(config_path)
    
    try:
        print("Проверка списка подготовленных видео...")
        videos = pipeline.list_prepared_videos()
        print(f"Найдено подготовленных видео: {len(videos)}")
        if len(videos) == 0:
            print("Ошибка: подготовленные видео не найдены в data/videos/")
            sys.exit(1)
            
        print("Выполнение тестового поиска по индексу Qdrant...")
        query = "роблокс"
        decomposition, candidates = pipeline.search(query)
        
        print(f"Поиск завершен. Найдено кандидатов: {len(candidates)}")
        for idx, c in enumerate(candidates, 1):
            print(f"[{idx}] Видео: {Path(c.video_file).name} | Score: {c.score:.4f} | Интервал: {c.start:.1f}s - {c.end:.1f}s")
            
        if not candidates:
            print("Внимание: Поиск вернул 0 результатов, но индекс должен быть не пустым.")
            
        print("=== Smoke Test PASSED успешно! ===")
        
    except Exception as e:
        print(f"Ошибка при прохождении Smoke-теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        pipeline.close()

if __name__ == "__main__":
    run_smoke_test()
