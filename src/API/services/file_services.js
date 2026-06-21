import { baseUrl } from "../apiConsts";

/**
 * ������� ��� ������ API � ��������� ������ �� ZIP-�����.
 *
 * @returns {Promise<string>} URL ��� ���������� ZIP-�����.
 * @throws {Error} ���� ������ �� ������ ��� ������ ������ ������.
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
 * ������� ��� �������� ������ CSV �� ������.
 *
 * @param {File[]} files - ������ ������ (CSV).
 * @returns {Promise<string>} ����� ������� � ������ ������.
 * @throws {Error} ���� ������ ���������� �������.
 */
export async function uploadCSVFiles(files) {
  const API_ENDPOINT = `${baseUrl}/upload_csv/`;

  // ������� ������ FormData ��� �������� ������
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      body: formData, // �������� �����
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
