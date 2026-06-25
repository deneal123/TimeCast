import { Instance } from "../instance";

export async function fetchZipUrl() {
  const data = await Instance.get("/get_zip");
  if (!data.url) throw new Error("Response does not contain a valid URL.");
  return data.url;
}

export async function uploadCSVFiles(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  const data = await Instance.post("/upload_csv/", formData);
  return data.message || "Files uploaded successfully!";
}
