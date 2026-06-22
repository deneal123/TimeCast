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

/**
 * Обобщённый ряд: POST /timeseries_inference/ — инференс на любом tidy-CSV.
 * Тело: { dataset: {source, ...}, inference: {dictseasonal, future_or_estimate} }.
 */
export async function sendTimeSeriesInference(requestData) {
  const API_ENDPOINT = `${baseUrl}/timeseries_inference/`;

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
    console.error("Error in timeseries inference request:", error);
    throw error;
  }
}

/** Обобщённый ряд: POST /timeseries_neiro_graduate/ — обучение нейросети на любом tidy-CSV. */
export async function sendTimeSeriesNeiroGraduate(requestData) {
  const API_ENDPOINT = `${baseUrl}/timeseries_neiro_graduate/`;

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestData),
    });
    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error in timeseries neiro graduate request:", error);
    throw error;
  }
}

/** Обобщённый ряд: POST /timeseries_neiro_inference/ — инференс нейросети на любом tidy-CSV. */
export async function sendTimeSeriesNeiroInference(requestData) {
  const API_ENDPOINT = `${baseUrl}/timeseries_neiro_inference/`;

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestData),
    });
    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error in timeseries neiro inference request:", error);
    throw error;
  }
}
