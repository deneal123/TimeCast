import { Instance } from "../instance";

export const getTask = (taskId) => Instance.get(`/tasks/${taskId}`);
export const listTasks = () => Instance.get("/tasks/");

export const queueClassicGraduate = (requestData) =>
  Instance.post("/classic_graduate/queue/", requestData);
export const queueNeiroGraduate = (requestData) =>
  Instance.post("/neiro_graduate/queue/", requestData);
export const queueTimeSeriesGraduate = (requestData) =>
  Instance.post("/timeseries_graduate/queue/", requestData);
export const queueTimeSeriesNeiroGraduate = (requestData) =>
  Instance.post("/timeseries_neiro_graduate/queue/", requestData);
