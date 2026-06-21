import { baseUrl } from "../apiConsts";

/**
 * ������� ��� ������ ��������� /season_analytic/.
 *
 * @param {Object} requestData - ������ Query � ������� JSON.
 * @returns {Promise<Object>} ����� �� �������.
 * @throws {Error} ���� ������ ���������� �������.
 */
export async function sendSeasonAnalytic(requestData) {
  const API_ENDPOINT = `${baseUrl}/season_analytic/`;

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json", // ���������, ��� �������� JSON
      },
      body: JSON.stringify(requestData), // ����������� ������ � ������ JSON
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
