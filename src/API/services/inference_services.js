import { baseUrl } from "../apiConsts";

/**
 * API ��� ������ ��������� /classic_inference/
 *
 * @param {Object} requestData - ������ ��� ��������� ������������ �������.
 * @returns {Promise<Object>} ����� �� �������.
 * @throws {Error} ���� ������ ���������� �������.
 */
export async function sendClassicInference(requestData) {
  const API_ENDPOINT = `${baseUrl}/classic_inference/`;

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
    console.error("Error in classic inference request:", error);
    throw error;
  }
}

/**
 * API ��� ������ ��������� /neiro_inference/
 *
 * @param {Object} requestData - ������ ��� ��������� ������������ �������.
 * @returns {Promise<Object>} ����� �� �������.
 * @throws {Error} ���� ������ ���������� �������.
 */
export async function sendNeiroInference(requestData) {
  const API_ENDPOINT = `${baseUrl}/neiro_inference/`;

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
    console.error("Error in neiro inference request:", error);
    throw error;
  }
}
