# Frontend — TimeCast

SPA на **React 18**, собранная через Create React App (пакет `timecast-frontend`).
UI — **Chakra UI**. Назначение: интерфейс к бэкенду TimeCast — формирование
JSON-запроса, отправка, просмотр живого лога обучения/инференса, **визуализация
прогноза (Plotly)** и скачивание результатов.

## Технологический стек

| Категория | Библиотека | Примечание |
|-----------|-----------|------------|
| Ядро | React 18.2 | хуки, без классов |
| Роутинг | react-router-dom 6 | `createHashRouter` (хэш-роутинг) |
| UI | Chakra UI 2.8 + Emotion | кастомная тема `theme.js` |
| Графики | plotly.js + react-plotly.js | дашборд прогноза (`ForecastChart.jsx`, фабрика на `plotly.js-dist-min`) |
| HTTP | axios + нативный `fetch` | базовый URL из `REACT_APP_API_BASE_URL` |
| Логи | EventSource (SSE) | `apiLogStreamComponent.jsx` |
| Markdown | react-markdown + rehype-raw + react-syntax-highlighter | страница документации |
| Шрифты | @fontsource/montserrat | |

> Неиспользуемые зависимости (formik, yup, tesseract.js, react-pdf, react-select,
> react-search-autocomplete, js-cookie) удалены. `framer-motion` оставлен — это
> обязательная peer-зависимость Chakra UI.

## Роутинг и страницы

Хэш-роутинг, `/` → редирект на `/main` ([App.js](../frontend/src/App.js)).

| Путь | Страница | Назначение |
|------|----------|------------|
| `/main` | `main_page.jsx` | Лендинг: вертикальное меню + картинка |
| `/query` | `query_page.jsx` | Основной экран: textarea запроса, textarea ответа/логов, загрузка CSV, кнопки действий |
| `/documentation` | `documentation_page.jsx` | Документация по классам/методам бэкенда (markdown захардкожен строками) |
| `/*` | `notfound_page.jsx` | 404 (пустой контейнер) |

`Layout.jsx` оборачивает страницы в `Header` + `Footer` и условно показывает пункты
меню в зависимости от текущего пути.

## API-слой

- **Базовый URL захардкожен**: `http://localhost:8080/server`
  ([apiConsts.js](../frontend/src/API/apiConsts.js)) — без `.env`.
- Сервисы (`API/services/*.js`) вызывают бэкенд через `fetch`:

| Файл | Эндпоинты |
|------|-----------|
| `file_services.js` | `GET /get_zip`, `POST /upload_csv/` |
| `graduate_services.js` | `POST /classic_graduate/`, `POST /neiro_graduate/` |
| `inference_services.js` | `POST /classic_inference/`, `POST /neiro_inference/` |
| `season_analytic_services.js` | `POST /season_analytic/` |

- **Выбор эндпоинта — по структуре JSON** на стороне фронта
  ([query_page.jsx](../frontend/src/pages/query_page.jsx)): наличие ключей
  `inference` / `graduate` / `proccess` и вложенных полей определяет, какой сервис
  вызвать. Хрупко: опечатка в JSON ведёт не туда.
- **Стрим логов**: `LogStreamComponent` открывает `EventSource` на `/stream-logs` и
  дописывает строки в `responseText`.

## Компоненты

| Компонент | Роль |
|-----------|------|
| `MenuComponent` | Меню с навигацией (`useNavigate`) |
| `MenuActiveComponent` | Меню без навигации — только колбэки действий (Send CSV / Send Query / Load Zip) |
| `Header` / `footer` | Шапка с логотипом, подвал |
| `MarkdownRenderer` | Рендер markdown + подсветка кода |
| `LineBarComponent` | Статичная панель-контрол (не график!) |
| `DocFieldComponent` | Декоративный контейнер документации |

## Состояние и данные

- Состояние **только локальное** (`useState`), без Redux/Context. Местами
  prop-drilling (карты колбэков в `QueryPage`).
- Хук `useFetching` — минимальная обёртка try/catch, почти не используется.
- `window_dimensions` / `useDimensions` — адаптивная вёрстка по ширине окна.

## Аутентификация

- `common/private_Routes.jsx` объявлен, но **не подключён** в роутинге — все
  страницы публичны.
- `CookiesProvider` инициализирован в `index.js`, но логина/токенов нет.

## Сборка и запуск

Стандартный CRA: `npm start` (dev, :3000), `npm run build`, `npm test`
(тестов в `src/` нет).

См. также [observations.md](observations.md) — техдолг фронтенда.
