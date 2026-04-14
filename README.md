# Video-RAG Research

Исследовательский проект для репликации и адаптации `Video-RAG` под русскоязычные ролики из `lera_materials/`. Кодовая база построена вокруг модульного пайплайна: разбор вопроса, извлечение вспомогательных текстов из видео, индексирование и поиск по временным интервалам.

Базовый пайплайн:

1. Подготовить локальный набор видео из `lera_materials/`, включая zip-архивы.
2. Разобрать пользовательский вопрос через LVLM в формате статьи: `R = {R_asr, R_det, R_type}`.
3. Построить три независимых набора вспомогательных текстов:
   `ASR` через `Whisper`, `OCR` через `EasyOCR`, `DET` через `BLIP + spaCy`.
4. Проиндексировать результаты по модальностям в локальном `Qdrant`.
5. На запросе выполнить поиск по модальностям, склеить совпадения по времени и отфильтровать их по порогу.
6. Передать найденные вспомогательные тексты и видеоклип верхнего интервала в LVLM для финального ответа.

## Почему конфиг по умолчанию заточен под `mps`

- `PyTorch` рекомендует использовать устройство `mps` на Apple Silicon и допускает CPU fallback через `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- В актуальной документации `PyTorch 2.11` отдельно описаны watermark-параметры MPS allocator; здесь они выставлены консервативно, чтобы не упираться в память на `M4 Pro`.
- Для компактных моделей на Apple Silicon у `Transformers` уже есть отдельная ветка `Metal`-квантизации, но в этом репозитории по умолчанию остаётся обычный режим, чтобы не усложнять воспроизводимость.

Источники:

- [Video-RAG, arXiv:2411.13093](https://arxiv.org/abs/2411.13093)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)
- [Transformers Metal quantization](https://huggingface.co/docs/transformers/main/quantization/metal)

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Системные зависимости:

- `ffmpeg` нужен для разбора видео и аудио.
- На macOS: `brew install ffmpeg`.
- Для быстрого поиска без загрузки embedding-модели в каждом процессе можно держать локальный TEI-сервер:

```bash
brew install text-embeddings-inference
text-embeddings-router --model-id Qwen/Qwen3-Embedding-0.6B --port 8080
```

По умолчанию `configs/config.yaml` использует `indexing.embedding_backend: tei` и endpoint `http://127.0.0.1:8080`. Если TEI не запущен, временно переключи `embedding_backend` на `local`.

Gemini для LVLM-стадий:

```bash
cp .env.example .env
```

В `.env` укажи ключи через запятую:

```env
GOOGLE_API_KEYS=key_1,key_2
```

Ключи перебираются по очереди, модели тоже. По умолчанию используются `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-2.5-flash`. Для `Flash` включён минимальный режим размышления, чтобы снизить задержку.

Подготовить датасет из материалов Леры:

```bash
python -m scripts.prepare_dataset --config configs/config.yaml
```

Построить индексы:

```bash
python -m scripts.build_index --config configs/config.yaml --recreate
```

Проверить поиск:

```bash
python -m scripts.search --config configs/config.yaml "Где в видео говорят про роблокс?"
```

Получить финальный ответ по схеме статьи:

```bash
python -m scripts.ask --config configs/config.yaml "Где в видео говорят про роблокс?"
```

Если нужно увидеть, какие фрагменты были переданы в финальную модель:

```bash
python -m scripts.ask --config configs/config.yaml --show-context "Где в видео говорят про роблокс?"
```

## Структура

```text
configs/config.yaml        основной локальный конфиг
lera_materials/            материалы Леры, включая pdf и zip с видео
scripts/prepare_dataset.py подготовка видео из материалов
scripts/build_index.py     запуск модулей и сборка индексов
scripts/search.py          поиск по готовым индексам
scripts/ask.py             полный запрос с финальной LVLM-генерацией
src/generation/            Gemini-клиент, query decouple и answer generation
src/modules/               ASR, OCR, DET и query decouple
src/retrieval/             embedder, qdrant, reranker
src/pipeline.py            оркестратор нового пайплайна
```

## Особенности

- Пайплайн строится вокруг трёх модальностей статьи: `ASR`, `OCR` и `DET`.
- Индекс хранится по модальностям, а результаты затем сшиваются по времени.
- `scripts.search` нужен для отладки retrieval-части, `scripts.ask` запускает полный цикл с финальным LVLM-ответом.
- Конфигурация по умолчанию рассчитана на `mps`, при этом `cuda` остаётся рабочей опцией.
