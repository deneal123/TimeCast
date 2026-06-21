import axios from "axios";
import { baseUrl } from "./apiConsts";

export const Instance = axios.create({
  baseURL: baseUrl,
});
