// Базовый URL backend API.
// Берётся из переменной окружения REACT_APP_API_BASE_URL (см. .env.example),
// со значением по умолчанию для локальной разработки.
export const baseUrl =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:8080/server";
