import React, { useState } from "react";
import {
  VStack,
  HStack,
  Input,
  Select,
  Button,
  Text,
  Box,
} from "@chakra-ui/react";

/**
 * Конструктор запроса для ПРОИЗВОЛЬНОГО временного ряда (tidy CSV).
 * Собирает JSON в форме, которую понимает query_page (dataset.source + graduate/inference),
 * и кладёт его в текстовое поле запроса через onBuild(jsonString).
 */
const fieldStyle = {
  bg: "#2D2D2D",
  color: "#FFFFFF",
  borderColor: "#FF0032",
  size: "sm",
};

const IFFT_DEFAULT = {
  depth: 6,
  dim: 256,
  dim_head: 64,
  heads: 8,
  num_tokens_per_variate: 1,
  use_reversible_instance_norm: true,
};

const GenericSeriesForm = ({ onBuild }) => {
  const [source, setSource] = useState("series.csv");
  const [seriesIdCol, setSeriesIdCol] = useState("id");
  const [timeCol, setTimeCol] = useState("");
  const [targetCol, setTargetCol] = useState("");
  const [featureCols, setFeatureCols] = useState("");
  const [mode, setMode] = useState("graduate"); // graduate | inference | decompose
  const [model, setModel] = useState("classic"); // classic | neiro
  const [seasonal, setSeasonal] = useState("week:7, month:30, quater:90");
  const [futureOrEstimate, setFutureOrEstimate] = useState("estimate");
  const [seqLen, setSeqLen] = useState("30");

  const parseSeasonal = () => {
    const out = {};
    seasonal.split(",").forEach((pair) => {
      const [k, v] = pair.split(":").map((s) => s && s.trim());
      if (k && v && !Number.isNaN(Number(v))) out[k] = Number(v);
    });
    return out;
  };

  const build = () => {
    const dataset = { source: source.trim() };
    if (seriesIdCol.trim()) dataset.series_id_col = seriesIdCol.trim();
    if (timeCol.trim()) dataset.time_col = timeCol.trim();
    if (targetCol.trim()) dataset.target_col = targetCol.trim();
    const feats = featureCols
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (feats.length) dataset.feature_cols = feats;

    const dictseasonal = parseSeasonal();
    const isNeiro = model === "neiro";

    let payload;
    if (mode === "decompose") {
      payload = { dataset, proccess: { dictdecompose: dictseasonal } };
    } else if (mode === "graduate") {
      const graduate = { dictseasonal };
      if (isNeiro) {
        graduate.dictmodels = { IFFT: { ...IFFT_DEFAULT } };
        graduate.seq_len = Number(seqLen) || 30;
        graduate.use_device = "cuda";
      } else {
        graduate.models_params = { AUTOARIMA: [3, 3, 0, 0, 1, 1, "week"] };
      }
      payload = { dataset, graduate };
    } else {
      const inference = { dictseasonal, future_or_estimate: futureOrEstimate };
      if (isNeiro) {
        inference.dictmodels = { IFFT: { ...IFFT_DEFAULT } };
        inference.seq_len = Number(seqLen) || 30;
        inference.use_device = "cuda";
      }
      payload = { dataset, inference };
    }

    onBuild(JSON.stringify(payload, null, 2));
  };

  return (
    <VStack align="stretch" spacing={2} mb={4}>
      <Text fontSize="16px" fontWeight="bold" color="#FFFFFF">
        Generic series builder
      </Text>
      <Input
        {...fieldStyle}
        placeholder="source CSV (e.g. series.csv)"
        value={source}
        onChange={(e) => setSource(e.target.value)}
      />
      <HStack>
        <Input
          {...fieldStyle}
          placeholder="series_id_col"
          value={seriesIdCol}
          onChange={(e) => setSeriesIdCol(e.target.value)}
        />
        <Input
          {...fieldStyle}
          placeholder="time_col (optional)"
          value={timeCol}
          onChange={(e) => setTimeCol(e.target.value)}
        />
        <Input
          {...fieldStyle}
          placeholder="target_col (optional)"
          value={targetCol}
          onChange={(e) => setTargetCol(e.target.value)}
        />
      </HStack>
      <Input
        {...fieldStyle}
        placeholder="feature_cols (comma-separated, optional)"
        value={featureCols}
        onChange={(e) => setFeatureCols(e.target.value)}
      />
      <HStack>
        <Select {...fieldStyle} value={mode} onChange={(e) => setMode(e.target.value)}>
          <option style={{ color: "#000" }} value="graduate">
            train
          </option>
          <option style={{ color: "#000" }} value="inference">
            inference
          </option>
          <option style={{ color: "#000" }} value="decompose">
            decompose
          </option>
        </Select>
        {mode !== "decompose" && (
          <Select {...fieldStyle} value={model} onChange={(e) => setModel(e.target.value)}>
            <option style={{ color: "#000" }} value="classic">
              classic
            </option>
            <option style={{ color: "#000" }} value="neiro">
              neiro
            </option>
          </Select>
        )}
      </HStack>
      <Input
        {...fieldStyle}
        placeholder="seasonal (name:horizon, ...)"
        value={seasonal}
        onChange={(e) => setSeasonal(e.target.value)}
      />
      {mode !== "decompose" && (
        <HStack>
          {mode === "inference" && (
            <Select
              {...fieldStyle}
              value={futureOrEstimate}
              onChange={(e) => setFutureOrEstimate(e.target.value)}
            >
              <option style={{ color: "#000" }} value="estimate">
                estimate
              </option>
              <option style={{ color: "#000" }} value="future">
                future
              </option>
            </Select>
          )}
          {model === "neiro" && (
            <Input
              {...fieldStyle}
              placeholder="seq_len"
              value={seqLen}
              onChange={(e) => setSeqLen(e.target.value)}
            />
          )}
        </HStack>
      )}
      <Box>
        <Button
          size="sm"
          bg="#FF0032"
          color="#FFFFFF"
          _hover={{ bg: "#cc0028" }}
          onClick={build}
        >
          Build Request
        </Button>
      </Box>
    </VStack>
  );
};

export default GenericSeriesForm;
