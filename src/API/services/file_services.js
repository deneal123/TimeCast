import { baseUrl } from "../apiConsts";
import { Instance } from "../instance";

export async function fetchZipUrl() {
  const data = await Instance.get("/get_zip");
  if (!data.filename) throw new Error("Response does not contain filename.");
  return `${baseUrl}/public/zip/${data.filename}`;
}

export async function uploadCSVFiles(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  const data = await Instance.post("/upload_csv/", formData);
  return data.message || "Files uploaded successfully!";
}
