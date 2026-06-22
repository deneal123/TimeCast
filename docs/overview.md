# TimeCast — Обзор проекта

> **Северная звезда:** платформа для обучения моделей предсказания **любых**
> временных рядов с фичами и инференса на них (розничные продажи — частный случай).

TimeCast — учебный/исследовательский проект (ВШЭ × МТС, временные ряды). Состоит из
двух частей:

| Часть | Стек | Назначение |
|-------|------|------------|
| **timecast** | Python-пакет (src-layout): PyTorch, sktime, statsmodels, DuckDB | Переиспользуемая библиотека — ядро: обучение, инференс, сезонная аналитика |
| **backend** | FastAPI поверх `timecast` | Тонкий HTTP/SSE-слой: маршруты, валидация, стрим логов, структурные результаты |
| **frontend** | React 18, Chakra UI, Plotly | Веб-дашборд: запросы, live-лог, графики прогноза, скачивание результатов |

## Доменная модель в одном абзаце

**Обобщённый режим (основной):** на вход подаётся один tidy-CSV (`time, target,
[series_id], feature_*`). Данные раскладываются по `series_id`, и для каждого ряда
строится прогноз. Не привязан к домену — подойдёт любой временной ряд с фичами.

**Розничный режим (частный случай):** три CSV (`shop_sales.csv`,
`shop_sales_dates.csv`, `shop_sales_prices.csv`) — продажи, календарь дат/событий и
цены. Склеиваются через DuckDB, раскладываются по `store_id → item_id`, строится
прогноз спроса. Формат распознаётся автоматически.

Поддерживаются два семейства моделей:

- **Classic** — статистические модели (AutoARIMA, AutoREG, AutoETS, Prophet, TBATS из `sktime`).
- **Neiro** — нейросеть `iTransformer` (PyTorch) с кастомным `CustomLoss`.

Плюс **SeasonAnalytic** — сезонная декомпозиция ряда (тренд / сезонность / остаток)
без обучения. Готовые веса моделей подгружаются с HuggingFace
(`GrafTrahula/STORE_CLASSIC`, `GrafTrahula/STORE_NEIRO`).

## Поток данных (end-to-end)

```
React (QueryPage)                 FastAPI (/server)              Library
  форма/JSON     ──POST──►  /timeseries_graduate/  ──► service ──► train_classic_series   (обобщённый ряд)
                            /timeseries_inference/                 infer_classic_series
                            /timeseries_neiro_graduate/            train_neiro_series
                            /timeseries_neiro_inference/           infer_neiro_series
                            /classic_graduate/      ──►  service ──► ClassicGraduatePipeline  (retail)
                            /neiro_graduate/                      NeiroGraduatePipeline
                            /classic_inference/                   ClassicInferencePipeline
                            /neiro_inference/                     NeiroInferencePipeline
                            /season_analytic/                     SeasonAnalyticPipeline
  EventSource    ◄──SSE───  /stream-logs   (живой лог обучения/инференса)
  upload CSV     ──POST──►  /upload_csv/
  download zip   ◄──GET───  /get_zip/     (графики + результаты архивом)
```

## Карта документации

- [architecture-target.md](architecture-target.md) — **целевая архитектура** (вектор: библиотека-ядро + дашборд).
- [backend.md](backend.md) — архитектура бэкенда, слои, API, ML-пайплайны.
- [frontend.md](frontend.md) — архитектура фронтенда, роутинг, API-слой.
- [observations.md](observations.md) — наблюдения по качеству кода и техдолгу.
- [../TODO.md](../TODO.md) — этапы рефакторинга под целевую архитектуру.

> ⚠️ Документ составлен по состоянию кода на 2026-06-20. Версии и факты сверены с
> исходниками; спорные места помечены.
