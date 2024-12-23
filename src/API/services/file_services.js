import { baseUrl } from "../apiConsts";

/**
 * Функция для вызова API и получения ссылки на ZIP-архив.
 * 
 * @returns {Promise<string>} URL для скачивания ZIP-файла.
 * @throws {Error} Если запрос не удался или сервер вернул ошибку.
 */
export async function fetchZipUrl() {
  
    const API_ENDPOINT = `${baseUrl}/get_zip`;

    try {
        const response = await fetch(API_ENDPOINT, {
            method: "GET",
            headers: { "Content-Type": "application/json" },
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();

        if (!data.url) {
            throw new Error("Response does not contain a valid URL.");
        }

        return data.url;
    } catch (error) {
        console.error("Error while fetching the ZIP URL:", error);
        throw error;
    }
}


/**
 * Функция для отправки файлов CSV на сервер.
 *
 * @param {File[]} files - Массив файлов (CSV).
 * @returns {Promise<string>} Ответ сервера в случае успеха.
 * @throws {Error} Если запрос завершился ошибкой.
 */
export async function uploadCSVFiles(files) {

    const API_ENDPOINT = `${baseUrl}/upload_csv/`;

    // Создаем объект FormData для отправки файлов
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    try {
        const response = await fetch(API_ENDPOINT, {
            method: "POST",
            body: formData, // Передаем файлы
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        return data.message || "Files uploaded successfully!";
    } catch (error) {
        console.error("Error while uploading CSV files:", error);
        throw error;
    }
}