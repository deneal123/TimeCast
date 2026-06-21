# TimeCast — Обзор проекта

> **Северная звезда:** платформа для обучения моделей предсказания временных рядов
> и инференса на них (продажи в розничной сети).

TimeCast — учебный/исследовательский проект (ВШЭ × МТС, временные ряды). Состоит из
двух частей:

| Часть | Стек | Назначение |
|-------|------|------------|
| **timecast** | Python-пакет (src-layout): PyTorch, sktime, statsmodels, DuckDB | Переиспользуемая библиотека — ядро: обучение, инференс, сезонная аналитика |
| **backend** | FastAPI поверх `timecast` | Тонкий HTTP/SSE-слой: маршруты, валидация, стрим логов, структурные результаты |
| **frontend** | React 18, Chakra UI, Plotly | Веб-дашборд: запросы, live-лог, графики прогноза, скачивание результатов |

## Доменная модель в одном абзаце

На вход подаются три CSV (`shop_sales.csv`, `shop_sales_dates.csv`,
`shop_sales_prices.csv`) — продажи, календарь дат/событий и цены. Данные
склеиваются через DuckDB, раскладываются по `store_id → item_id`, и для каждого
товара строится прогноз спроса. Поддерживаются два семейства моделей:

- **Classic** — статистические модели (AutoARIMA, AutoREG, AutoETS, Prophet, TBATS из `sktime`).
- **Neiro** — нейросеть `iTransformer` (PyTorch) с кастомным `CustomLoss`.

Плюс **SeasonAnalytic** — сезонная декомпозиция ряда (тренд / сезонность / остаток)
без обучения. Готовые веса моделей подгружаются с HuggingFace
(`GrafTrahula/STORE_CLASSIC`, `GrafTrahula/STORE_NEIRO`).

## Поток данных (end-to-end)

```
React (QueryPage)                 FastAPI (/server)              Library
  textarea JSON  ──POST──►  /classic_graduate/  ──►  service ──► ClassicGraduatePipeline
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
