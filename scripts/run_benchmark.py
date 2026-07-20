from __future__ import annotations

import time
import argparse
from pathlib import Path
from src.pipeline import VideoRAGPipeline
from src.models import QueryDecomposition

# Новые тестовые запросы, разделенные по категориям
BENCHMARK_TASKS = [
    # 1. Речевые запросы (содержатся в ASR)
    {
        "query": "Где говорят про роблокс?",
        "expected_video": "roblox",
        "category": "speech"
    },
    {
        "query": "Рецепт итальянской пасты карбонара с панчеттой и желтками",
        "expected_video": "carbonara",
        "category": "speech"
    },
    {
        "query": "Как правильно завязать галстук пошагово",
        "expected_video": "galstuk",
        "category": "speech"
    },
    {
        "query": "Как приготовить домашнюю пиццу в духовке",
        "expected_video": "pizza",
        "category": "speech"
    },
    {
        "query": "Как приготовить блины на молоке",
        "expected_video": "blini",
        "category": "speech"
    },
    # 2. OCR-запросы (текст на экране, отсутствует в ASR)
    {
        "query": "Вино марки Duckhorn Vineyards",
        "expected_video": "vino",
        "category": "ocr"
    },
    {
        "query": "Продюсер Василий Нефедкин в титрах",
        "expected_video": "obuv",
        "category": "ocr"
    },
    # 3. Визуальные запросы (объекты в кадре, отсутствуют в ASR)
    {
        "query": "Человек стоит перед самолетом",
        "expected_video": "bomber",
        "category": "visual"
    },
    {
        "query": "Люди стоят около танка на заднем плане",
        "expected_video": "bomber",
        "category": "visual"
    },
    {
        "query": "Бутылки вина на столе",
        "expected_video": "vino",
        "category": "visual"
    }
]

def run_evaluation(pipeline: VideoRAGPipeline, mode: str) -> dict:
    if mode == "asr-only":
        pipeline.cfg["ocr"]["enabled"] = False
        pipeline.cfg["det"]["enabled"] = False
    elif mode == "multimodal":
        pipeline.cfg["ocr"]["enabled"] = True
        pipeline.cfg["det"]["enabled"] = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    hits_top1 = 0
    hits_top3 = 0
    total_latency = 0.0
    total_qdrant_latency = 0.0
    
    # Категорийная статистика
    category_stats = {
        "speech": {"total": 0, "top1": 0, "top3": 0},
        "ocr": {"total": 0, "top1": 0, "top3": 0},
        "visual": {"total": 0, "top1": 0, "top3": 0}
    }
    
    results = []

    print(f"\n=== Запуск бенчмарка в режиме: {mode.upper()} ===")

    for task in BENCHMARK_TASKS:
        query = task["query"]
        expected = task["expected_video"]
        category = task["category"]
        
        category_stats[category]["total"] += 1

        t_start = time.perf_counter()
        
        # Шаг 1: Декомпозиция (Gemini)
        decomposition = QueryDecomposition(original_query=query, asr_query=query, det_queries=[], det_mode="relation")
        query_decoupler = None
        try:
            query_decoupler = pipeline._get_query_decoupler()
            if query_decoupler is not None:
                decomposition = query_decoupler.decouple(query)
        except Exception as e:
            print(f"Warning: Gemini decoupler failed ({e}). Using default decomposition.")
        
        # Шаг 2: Поиск в векторной базе (измеряем только локальное время без сети)
        t_qdrant_start = time.perf_counter()
        
        modality_queries = {
            "asr": pipeline._build_text_retrieval_query(query, decomposition.asr_query),
            "ocr": pipeline._build_text_retrieval_query(query, decomposition.asr_query),
            "det": pipeline._build_det_query(decomposition),
        }
        all_hits = []
        top_k = pipeline.cfg["search"].get("per_modality_top_k", 12)
        store = pipeline._get_store()
        embedder = pipeline._get_embedder()
        score_threshold = pipeline.cfg["search"].get("score_threshold")
        
        for modality in pipeline.enabled_modalities():
            modality_query = modality_queries.get(modality) or query
            if not modality_query.strip():
                continue
            query_vector = embedder.embed_query(modality_query)
            filter_payload = (
                {"det_type": pipeline._det_type_for_mode(decomposition.det_mode)}
                if modality == "det"
                else None
            )
            hits = store.search(modality, query_vector, top_k=top_k, filter_payload=filter_payload)
            if score_threshold is not None:
                hits = [hit for hit in hits if hit.score >= float(score_threshold)]
            all_hits.extend(hits)
            
        candidates = pipeline._merge_hits(all_hits)
        final_top_k = pipeline.cfg["search"].get("final_top_k", 5)
        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[:final_top_k]
        
        t_qdrant = time.perf_counter() - t_qdrant_start
        t_total = time.perf_counter() - t_start
        
        total_latency += t_total
        total_qdrant_latency += t_qdrant

        # Дедупликация кандидатов по уникальным видео для корректного Hit@K
        seen = set()
        top_videos = []
        for c in candidates:
            video_name = Path(c.video_file).stem.lower()
            if video_name not in seen:
                seen.add(video_name)
                top_videos.append(video_name)

        is_top1 = len(top_videos) > 0 and expected in top_videos[0]
        is_top3 = any(expected in v for v in top_videos[:3])

        if is_top1:
            hits_top1 += 1
            category_stats[category]["top1"] += 1
        if is_top3:
            hits_top3 += 1
            category_stats[category]["top3"] += 1

        status_str = "Hit@1" if is_top1 else ("Hit@3" if is_top3 else "Miss")
        print(f"[{category.upper()}] Запрос: \"{query}\" -> Ожидалось: {expected} | Топ-3: {top_videos[:3]} | {status_str} (Qdrant: {t_qdrant:.3f}s)")
        
        results.append({
            "query": query,
            "expected": expected,
            "category": category,
            "top_found": top_videos[:3],
            "t_qdrant": t_qdrant,
            "status": status_str
        })

    num_tasks = len(BENCHMARK_TASKS)
    metrics = {
        "hit_at_1": hits_top1 / num_tasks,
        "hit_at_3": hits_top3 / num_tasks,
        "avg_qdrant_latency": total_qdrant_latency / num_tasks,
        "avg_total_latency": total_latency / num_tasks,
        "category_stats": category_stats,
        "results": results
    }

    print(f"\n--- Метрики для {mode.upper()}:")
    print(f"Hit@1: {metrics['hit_at_1']:.2%}")
    print(f"Hit@3: {metrics['hit_at_3']:.2%}")
    print(f"Avg Qdrant Latency: {metrics['avg_qdrant_latency']:.3f}s")
    
    return metrics

def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк для сравнения ASR-only и гибридного мультимодального поиска")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="benchmark_results.md", help="Путь для записи отчета в формате Markdown")
    args = parser.parse_args()

    pipeline = VideoRAGPipeline(args.config)
    try:
        # 1. Warm-up
        print("Warm-up...")
        pipeline.search("тест")
        
        # 2. ASR-only
        asr_metrics = run_evaluation(pipeline, "asr-only")
        
        # Сброс пайплайна
        pipeline.close()
        pipeline = VideoRAGPipeline(args.config)
        pipeline.search("тест")
        
        # 3. Multimodal
        mm_metrics = run_evaluation(pipeline, "multimodal")
    finally:
        pipeline.close()

    # Запись результатов в markdown файл
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Расчет категорийных метрик для динамических выводов
    def get_cat_hit1(metrics, cat):
        stats = metrics["category_stats"][cat]
        return stats["top1"] / stats["total"] if stats["total"] > 0 else 0

    asr_speech_hit1 = get_cat_hit1(asr_metrics, "speech")
    mm_speech_hit1 = get_cat_hit1(mm_metrics, "speech")

    asr_ocr_total = asr_metrics["category_stats"]["ocr"]["total"]
    asr_visual_total = asr_metrics["category_stats"]["visual"]["total"]
    ocr_visual_total = asr_ocr_total + asr_visual_total

    asr_ocr_top1 = asr_metrics["category_stats"]["ocr"]["top1"]
    asr_visual_top1 = asr_metrics["category_stats"]["visual"]["top1"]
    asr_ocr_visual_hit1 = (asr_ocr_top1 + asr_visual_top1) / ocr_visual_total if ocr_visual_total > 0 else 0

    mm_ocr_top1 = mm_metrics["category_stats"]["ocr"]["top1"]
    mm_visual_top1 = mm_metrics["category_stats"]["visual"]["top1"]
    mm_ocr_visual_hit1 = (mm_ocr_top1 + mm_visual_top1) / ocr_visual_total if ocr_visual_total > 0 else 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Результаты тестирования поиска по видео (Benchmark)\n\n")
        f.write("Сравнение стратегий поиска: **ASR-only** (только аудиодорожка) против **Multimodal** (ASR + OCR + DET).\n")
        f.write("Тестирование проводилось на демонстрационном датасете из 30 видео. Общее число запросов: 10.\n\n")
        
        f.write("## 1. Сводные метрики (уровень видеофайлов)\n\n")
        f.write("| Стратегия | Hit@1 | Hit@3 | Avg Qdrant Latency |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **ASR-only** | {asr_metrics['hit_at_1']:.2%} | {asr_metrics['hit_at_3']:.2%} | {asr_metrics['avg_qdrant_latency']:.3f}s |\n")
        f.write(f"| **Multimodal (Гибрид)** | {mm_metrics['hit_at_1']:.2%} | {mm_metrics['hit_at_3']:.2%} | {mm_metrics['avg_qdrant_latency']:.3f}s |\n\n")
        
        f.write("## 2. Метрики по категориям запросов (Hit@1)\n\n")
        f.write("| Подмножество запросов | Кол-во | ASR-only Hit@1 | Multimodal Hit@1 |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        # Подсчет категорий
        for cat in ["speech", "ocr", "visual"]:
            asr_stats = asr_metrics["category_stats"][cat]
            mm_stats = mm_metrics["category_stats"][cat]
            asr_hit1 = asr_stats["top1"] / asr_stats["total"] if asr_stats["total"] > 0 else 0
            mm_hit1 = mm_stats["top1"] / mm_stats["total"] if mm_stats["total"] > 0 else 0
            f.write(f"| `{cat.upper()}` (Речевые/OCR/Визуальные) | {asr_stats['total']} | {asr_hit1:.2%} | {mm_hit1:.2%} |\n")
            
        f.write("\n## 3. Детальные результаты по запросам\n\n")
        f.write("| Запрос | Категория | Ожидаемое | ASR-only Топ-3 | Multimodal Топ-3 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        for i in range(len(BENCHMARK_TASKS)):
            q = BENCHMARK_TASKS[i]["query"]
            cat = BENCHMARK_TASKS[i]["category"]
            exp = BENCHMARK_TASKS[i]["expected_video"]
            asr_res = asr_metrics["results"][i]["top_found"]
            mm_res = mm_metrics["results"][i]["top_found"]
            f.write(f"| \"{q}\" | `{cat.upper()}` | `{exp}` | {asr_res} | {mm_res} |\n")
            
        f.write("\n## 4. Анализ и выводы\n\n")
        f.write(f"1. **Речевые запросы (`SPEECH`)**: В ситуациях, когда ключевое слово произносится спикером голосом, ASR-only Hit@1 составляет {asr_speech_hit1:.2%}, а Multimodal Hit@1 составляет {mm_speech_hit1:.2%}.\n")
        f.write(f"2. **Визуальные и OCR-запросы (`VISUAL` и `OCR`)**: ASR-only Hit@1 составляет {asr_ocr_visual_hit1:.2%}, в то время как Multimodal Hit@1 составляет {mm_ocr_visual_hit1:.2%}. Дополнительные OCR и DET признаки действительно содержат уникальную информацию, которая отсутствует в ASR (например, титры и визуальные объекты). Однако простое фиксированное слияние модальностей в текущей версии пайплайна снижает общую точность Hit@1 с {asr_metrics['hit_at_1']:.2%} до {mm_metrics['hit_at_1']:.2%}.\n")
        f.write(f"3. **Влияние на Latency**: Чистое время локального поиска в Qdrant составляет сотые доли секунды. Включение дополнительных индексов OCR и DET незначительно увеличивает время поиска (с {asr_metrics['avg_qdrant_latency']:.3f}s до {mm_metrics['avg_qdrant_latency']:.3f}s), что незаметно для пользователя и полностью компенсируется потенциалом роста полноты при правильной маршрутизации запросов.\n")
        f.write("4. **Дальнейшие шаги**: Для улучшения точности мультимодального поиска необходима динамическая классификация/маршрутизация запросов по типам (Query Routing) и динамическое взвешивание вкладов модальностей вместо фиксированного суммирования скоров.\n")

    print(f"\nРезультаты сохранены в: {output_path}")

if __name__ == "__main__":
    main()
