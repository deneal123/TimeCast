import { Instance } from "../instance";

export const sendClassicInference = (requestData) =>
  Instance.post("/classic_inference/", requestData);

export const sendNeiroInference = (requestData) =>
  Instance.post("/neiro_inference/", requestData);
