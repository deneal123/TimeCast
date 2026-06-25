import { Instance } from "../instance";

export const sendClassicGraduate = (requestData) =>
  Instance.post("/classic_graduate/", requestData);

export const sendNeiroGraduate = (requestData) =>
  Instance.post("/neiro_graduate/", requestData);

export const sendTimeSeriesGraduate = (requestData) =>
  Instance.post("/timeseries_graduate/", requestData);

export const sendTimeSeriesInference = (requestData) =>
  Instance.post("/timeseries_inference/", requestData);

export const sendTimeSeriesNeiroGraduate = (requestData) =>
  Instance.post("/timeseries_neiro_graduate/", requestData);

export const sendTimeSeriesNeiroInference = (requestData) =>
  Instance.post("/timeseries_neiro_inference/", requestData);
