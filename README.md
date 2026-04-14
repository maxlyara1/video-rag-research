# Video-RAG Research

Учебный исследовательский проект по репликации архитектуры Video-RAG для русскоязычных видео.

Цель проекта: по пользовательскому вопросу найти релевантные интервалы в наборе видео и сформировать ответ с опорой на речь, текст на экране, визуальные объекты и сам видеофрагмент.

## Ссылки

- [Статья Video-RAG, arXiv:2411.13093](https://arxiv.org/abs/2411.13093)
- [Официальный репозиторий Video-RAG](https://github.com/Leon1207/Video-RAG-master)

## Архитектура

Пайплайн повторяет три основные стадии Video-RAG:

1. `Query Decouple`: LVLM преобразует вопрос `Q` в `R = {R_asr, R_det, R_type}`.
2. `Auxiliary Text Generation & Retrieval`: строятся и ищутся вспомогательные тексты по трём модальностям.
3. `Integration & Generation`: найденные тексты `A_m`, вопрос и видеофрагмент передаются в LVLM для финального ответа.

```mermaid
%%{init: {"theme":"base","themeVariables":{"background":"#FFFFFF","primaryColor":"#EEF2FF","primaryBorderColor":"#4F46E5","primaryTextColor":"#0F172A","lineColor":"#64748B","fontFamily":"ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"},"flowchart":{"curve":"linear","nodeSpacing":42,"rankSpacing":56}}}%%
flowchart TD
    Q[Вопрос Q] --> D[Query Decouple: LVLM P,Q]
    D --> R[R_asr R_det R_type]

    V[Видео V] --> AUX[Auxiliary Text Generation]
    AUX --> DB[DB_asr DB_ocr DB_det]

    R --> RET[Retrieval по R и модальным базам]
    DB --> RET
    RET --> AM[A_m = Concat A_ocr A_asr A_det]
    RET --> INT[Найденный интервал в видео]
    INT --> FV[Видеофрагмент F_v]

    AM --> GEN[Integration and Generation: LVLM F_v, Concat A_m,Q]
    FV --> GEN
    GEN --> O[Ответ O]
```

Модальности:

- `ASR`: речь из видео через Whisper.
- `OCR`: текст на кадрах через EasyOCR.
- `DET`: scene graph по кадрам; записи разделены по `R_type`: `location`, `number`, `relation`.

Текущая реализация использует Gemini для LVLM-стадий, TEI для эмбеддингов и локальный Qdrant для индексов.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Для обработки видео используется `ffmpeg`:

```bash
brew install ffmpeg
```

`en_core_web_sm` используется только в `DET`: BLIP выдаёт англоязычные описания кадров, а spaCy разбирает их в scene graph. Русские `ASR` и `OCR` через spaCy не проходят.

## Настройка

Файл `.env` создаётся из примера:

```bash
cp .env.example .env
```

Минимальные переменные:

```env
GOOGLE_API_KEYS=key_1,key_2
EMBEDDING_BACKEND=tei
TEI_ENDPOINT=<tei-embedder-url>
```

Ключи Gemini задаются через запятую и перебираются по очереди. Адрес TEI не выводится в логах.

Основной конфиг находится в `configs/config.yaml`.

## TEI

На Mac с Apple Silicon `TEI` запускается на хосте, чтобы эмбеддер использовал `Metal`:

```bash
brew install text-embeddings-inference
```

Официальная инструкция для локального `Metal`: [Hugging Face TEI local Metal](https://huggingface.co/docs/text-embeddings-inference/en/local_metal).

```bash
text-embeddings-router \
  --model-id Qwen/Qwen3-Embedding-0.6B \
  --max-batch-tokens 4096 \
  --max-client-batch-size 64 \
  --auto-truncate true \
  --prometheus-port 9001 \
  --port 8080
```

Проверка:

```bash
curl http://127.0.0.1:8080/health
```

В `.env` указывается тот же адрес:

```env
EMBEDDING_BACKEND=tei
TEI_ENDPOINT=http://127.0.0.1:8080
```

В этом проекте используется один `TEI`-сервис: только для эмбеддингов.

## Данные

Исходные материалы кладутся в `lera_materials/`. Подготовленные видео сохраняются в `data/videos/`.

```bash
python -m scripts.prepare_dataset --config configs/config.yaml
```

## Индексация

```bash
python -m scripts.build_index --config configs/config.yaml --recreate
```

Команда извлекает или берёт из кэша `ASR`, `OCR`, `DET`, считает эмбеддинги и записывает индексы в Qdrant.

## Запросы

Только поиск по индексам:

```bash
python -m scripts.search --config configs/config.yaml "Где в видео говорят про роблокс?"
```

Полный Video-RAG с финальным ответом:

```bash
python -m scripts.ask --config configs/config.yaml "Где в видео говорят про роблокс?"
```

Вывод найденного контекста:

```bash
python -m scripts.ask --config configs/config.yaml --show-context "Где в видео говорят про роблокс?"
```

## Структура

```text
configs/config.yaml        основной конфиг
lera_materials/            исходные материалы
scripts/prepare_dataset.py подготовка видео
scripts/build_index.py     построение индексов
scripts/search.py          поиск по индексам
scripts/ask.py             поиск и финальный ответ
src/generation/            Gemini LVLM-стадии
src/modules/               ASR, OCR, DET
src/retrieval/             эмбеддинги, Qdrant, TEI-клиент
src/pipeline.py            общий пайплайн
```
