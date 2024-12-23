import { baseUrl } from "../apiConsts";

/**
 * Функция для вызова эндпоинта /season_analytic/.
 *
 * @param {Object} requestData - Данные Query в формате JSON.
 * @returns {Promise<Object>} Ответ от сервера.
 * @throws {Error} Если запрос завершился ошибкой.
 */
export async function sendSeasonAnalytic(requestData) {

    const API_ENDPOINT = `${baseUrl}/season_analytic/`;

    try {
        const response = await fetch(API_ENDPOINT, {
            method: "POST",
            headers: {
                "Content-Type": "application/json", // Указываем, что передаем JSON
            },
            body: JSON.stringify(requestData), // Преобразуем данные в строку JSON
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error while sending season analytic request:", error);
        throw error;
    }
}