import axios from "axios";
import { baseUrl } from "./apiConsts";

export const Instance = axios.create({
  baseURL: baseUrl,
});

// Unwrap data layer; normalise backend error detail (FastAPI returns [{msg, type}] or string).
Instance.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const detail = error.response?.data?.detail;
    const msg = Array.isArray(detail)
      ? detail.map((d) => d.msg).join("; ")
      : detail || error.message || "API error";
    return Promise.reject(new Error(msg));
  }
);
