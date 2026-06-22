import { baseUrl } from "../apiConsts";

/**
 * API ��� ������ ��������� /classic_graduate/
 *
 * @param {Object} requestData - ������ ��� ��������� ������������ �������.
 * @returns {Promise<Object>} ����� �� �������.
 * @throws {Error} ���� ������ ���������� �������.
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
 * API ��� ������ ��������� /neiro_graduate/
 *
 * @param {Object} requestData - ������ ��� ��������� ������������ �������.
 * @returns {Promise<Object>} ����� �� �������.
 * @throws {Error} ���� ������ ���������� �������.
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

/**
 * Обобщённый ряд: POST /timeseries_graduate/ — обучение на любом tidy-CSV.
 * Тело: { dataset: {source, ...}, graduate: {dictseasonal, models_params} }.
 */
export async function sendTimeSeriesGraduate(requestData) {
  const API_ENDPOINT = `${baseUrl}/timeseries_graduate/`;

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

    return await response.json();
  } catch (error) {
    console.error("Error in timeseries graduate request:", error);
    throw error;
  }
}
