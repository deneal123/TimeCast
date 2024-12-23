import { baseUrl } from "../apiConsts";


/**
 * API для вызова эндпоинта /classic_graduate/
 *
 * @param {Object} requestData - Данные для обработки классических моделей.
 * @returns {Promise<Object>} Ответ от сервера.
 * @throws {Error} Если запрос завершился ошибкой.
 */
export async function sendClassicGraduate(requestData) {
    const API_ENDPOINT = `${baseUrl}/classic_graduate/`;

    try {
        const response = await fetch(API_ENDPOINT, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestData),
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error in classic graduate request:", error);
        throw error;
    }
}

/**
 * API для вызова эндпоинта /neiro_graduate/
 *
 * @param {Object} requestData - Данные для обработки нейросетевых моделей.
 * @returns {Promise<Object>} Ответ от сервера.
 * @throws {Error} Если запрос завершился ошибкой.
 */
export async function sendNeiroGraduate(requestData) {
    const API_ENDPOINT = `${baseUrl}/neiro_graduate/`;

    try {
        const response = await fetch(API_ENDPOINT, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestData),
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error in neiro graduate request:", error);
        throw error;
    }
}
