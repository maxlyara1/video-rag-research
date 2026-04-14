# Video-RAG Research

Исследовательский проект для репликации и адаптации `Video-RAG` под русскоязычные ролики из `lera_materials/`. Кодовая база построена вокруг модульного пайплайна: разбор вопроса, извлечение вспомогательных текстов из видео, индексирование и поиск по временным интервалам.

Базовый пайплайн:

1. Подготовить локальный набор видео из `lera_materials/`, включая zip-архивы.
2. Разобрать пользовательский вопрос в формате статьи: `R = {R_asr, R_det, R_type}`.
3. Построить три независимых набора вспомогательных текстов:
   `ASR` через `Whisper`, `OCR` через `EasyOCR`, `DET` через `BLIP + spaCy`.
4. Проиндексировать результаты по модальностям в локальном `Qdrant`.
5. На запросе выполнить поиск по модальностям, склеить совпадения по времени и при необходимости переранжировать.

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

## Структура

```text
configs/config.yaml        основной локальный конфиг
lera_materials/            материалы Леры, включая pdf и zip с видео
scripts/prepare_dataset.py подготовка видео из материалов
scripts/build_index.py     запуск модулей и сборка индексов
scripts/search.py          поиск по готовым индексам
src/modules/               ASR, OCR, DET и query decouple
src/retrieval/             embedder, qdrant, reranker
src/pipeline.py            оркестратор нового пайплайна
```

## Особенности

- Пайплайн строится вокруг трёх модальностей статьи: `ASR`, `OCR` и `DET`.
- Индекс хранится по модальностям, а результаты затем сшиваются по времени.
- Конфигурация по умолчанию рассчитана на `mps`, при этом `cuda` остаётся рабочей опцией.
