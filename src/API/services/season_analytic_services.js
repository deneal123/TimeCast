import { Instance } from "../instance";

export const sendSeasonAnalytic = (requestData) =>
  Instance.post("/season_analytic/", requestData);
