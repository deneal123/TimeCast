import React, { useState, useRef } from "react";
import {
  VStack,
  HStack,
  Input,
  Select,
  Button,
  Text,
  Box,
  SimpleGrid,
  Collapse,
  Icon,
  Tooltip,
} from "@chakra-ui/react";
import { ChevronDownIcon, ChevronUpIcon, RepeatIcon } from "@chakra-ui/icons";

const INPUT_STYLE = {
  bg: "#0D1017",
  color: "#E8E8E8",
  border: "1px solid #2A2E36",
  borderRadius: "8px",
  size: "sm",
  _hover: { borderColor: "#444" },
  _focus: { borderColor: "#FF003288", boxShadow: "0 0 0 1px #FF003244" },
  _placeholder: { color: "#444" },
};

const SELECT_STYLE = {
  ...INPUT_STYLE,
  sx: {
    option: { background: "#1A1D21", color: "#E8E8E8" },
  },
};

const FieldLabel = ({ children }) => (
  <Text fontSize="11px" color="#555" fontWeight="600" letterSpacing="0.08em" mb="2px">
    {children}
  </Text>
);

const FORM_DEFAULTS = {
  source:          "series.csv",
  seriesIdCol:     "id",
  timeCol:         "",
  targetCol:       "",
  featureCols:     "",
  mode:            "graduate",
  model:           "classic",
  seasonal:        "week:7, month:30",
  futureOrEstimate:"estimate",
  seqLen:          "30",
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
  const [mode, setMode] = useState("graduate");
  const [model, setModel] = useState("classic");
  const [seasonal, setSeasonal] = useState("week:7, month:30");
  const [futureOrEstimate, setFutureOrEstimate] = useState("estimate");
  const [seqLen, setSeqLen] = useState("30");
  const [expanded,  setExpanded]  = useState(false);
  const [isDirty,   setIsDirty]   = useState(false);
  const didBuildRef = useRef(false);

  // Marks the form dirty once the user has built at least once.
  const markDirty = () => { if (didBuildRef.current) setIsDirty(true); };

  // Shorthand: returns an onChange handler that updates state and marks dirty.
  const field = (setter) => (e) => { setter(e.target.value); markDirty(); };

  const handleReset = () => {
    setSource(FORM_DEFAULTS.source);
    setSeriesIdCol(FORM_DEFAULTS.seriesIdCol);
    setTimeCol(FORM_DEFAULTS.timeCol);
    setTargetCol(FORM_DEFAULTS.targetCol);
    setFeatureCols(FORM_DEFAULTS.featureCols);
    setMode(FORM_DEFAULTS.mode);
    setModel(FORM_DEFAULTS.model);
    setSeasonal(FORM_DEFAULTS.seasonal);
    setFutureOrEstimate(FORM_DEFAULTS.futureOrEstimate);
    setSeqLen(FORM_DEFAULTS.seqLen);
    setExpanded(false);
    didBuildRef.current = false;
    setIsDirty(false);
  };

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
    const feats = featureCols.split(",").map((s) => s.trim()).filter(Boolean);
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
    didBuildRef.current = true;
    setIsDirty(false);
  };

  return (
    <VStack align="stretch" spacing={3} mb={3}>
      {/* Header row */}
      <HStack justify="space-between" align="center">
        <Text fontSize="12px" color="#555" fontWeight="600" letterSpacing="0.08em">
          КОНСТРУКТОР ЗАПРОСА
        </Text>
        <Button
          size="xs"
          variant="ghost"
          color="#555"
          _hover={{ color: "#AAA" }}
          rightIcon={<Icon as={expanded ? ChevronUpIcon : ChevronDownIcon} />}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Свернуть" : "Расширить"}
        </Button>
      </HStack>

      {/* Always visible: main fields */}
      <SimpleGrid columns={2} spacing={3}>
        <Box>
          <FieldLabel>Операция</FieldLabel>
          <Select {...SELECT_STYLE} value={mode} onChange={field(setMode)}>
            <option value="graduate">Обучение (train)</option>
            <option value="inference">Инференс</option>
            <option value="decompose">Декомпозиция</option>
          </Select>
        </Box>
        {mode !== "decompose" && (
          <Box>
            <FieldLabel>Модель</FieldLabel>
            <Select {...SELECT_STYLE} value={model} onChange={field(setModel)}>
              <option value="classic">Classic</option>
              <option value="neiro">Neural (iTransformer)</option>
            </Select>
          </Box>
        )}
      </SimpleGrid>

      <Box>
        <FieldLabel>Источник данных (CSV)</FieldLabel>
        <Input
          {...INPUT_STYLE}
          placeholder="series.csv"
          value={source}
          onChange={field(setSource)}
        />
      </Box>

      <Box>
        <FieldLabel>Сезонные периоды (name:horizon, ...)</FieldLabel>
        <Input
          {...INPUT_STYLE}
          placeholder="week:7, month:30"
          value={seasonal}
          onChange={field(setSeasonal)}
        />
      </Box>

      {/* Expanded fields */}
      <Collapse in={expanded} animateOpacity>
        <VStack align="stretch" spacing={3}>
          <SimpleGrid columns={3} spacing={3}>
            <Box>
              <FieldLabel>series_id_col</FieldLabel>
              <Input
                {...INPUT_STYLE}
                placeholder="id"
                value={seriesIdCol}
                onChange={field(setSeriesIdCol)}
              />
            </Box>
            <Box>
              <FieldLabel>time_col</FieldLabel>
              <Input
                {...INPUT_STYLE}
                placeholder="date"
                value={timeCol}
                onChange={field(setTimeCol)}
              />
            </Box>
            <Box>
              <FieldLabel>target_col</FieldLabel>
              <Input
                {...INPUT_STYLE}
                placeholder="value"
                value={targetCol}
                onChange={field(setTargetCol)}
              />
            </Box>
          </SimpleGrid>

          <Box>
            <FieldLabel>feature_cols (через запятую)</FieldLabel>
            <Input
              {...INPUT_STYLE}
              placeholder="price, promo"
              value={featureCols}
              onChange={field(setFeatureCols)}
            />
          </Box>

          {mode !== "decompose" && (
            <HStack spacing={3}>
              {mode === "inference" && (
                <Box flex={1}>
                  <FieldLabel>Режим</FieldLabel>
                  <Select
                    {...SELECT_STYLE}
                    value={futureOrEstimate}
                    onChange={field(setFutureOrEstimate)}
                  >
                    <option value="estimate">estimate (тест)</option>
                    <option value="future">future (прогноз)</option>
                  </Select>
                </Box>
              )}
              {model === "neiro" && (
                <Box flex={1}>
                  <FieldLabel>seq_len</FieldLabel>
                  <Input
                    {...INPUT_STYLE}
                    placeholder="30"
                    value={seqLen}
                    onChange={field(setSeqLen)}
                  />
                </Box>
              )}
            </HStack>
          )}
        </VStack>
      </Collapse>

      <HStack spacing={2}>
        <Tooltip
          label={isDirty ? "Форма изменилась — обновите JSON" : ""}
          placement="top"
          hasArrow
          isDisabled={!isDirty}
        >
          <Button
            size="sm"
            bg={isDirty ? "#FFBF00" : "#FF0032"}
            color={isDirty ? "#1A1D21" : "#FFFFFF"}
            _hover={{
              bg: isDirty ? "#E6AC00" : "#CC0028",
              transform: "translateY(-1px)",
            }}
            _active={{ transform: "translateY(0)" }}
            transition="all 0.15s"
            borderRadius="8px"
            onClick={build}
            px={5}
          >
            {isDirty ? "Обновить JSON" : "Сгенерировать JSON"}
          </Button>
        </Tooltip>
        <Tooltip label="Сбросить форму" placement="top" hasArrow>
          <Button
            size="sm"
            variant="ghost"
            color="#444"
            _hover={{ color: "#FF8888", bg: "#FF003212" }}
            borderRadius="8px"
            onClick={handleReset}
            px={2}
          >
            <Icon as={RepeatIcon} boxSize="14px" />
          </Button>
        </Tooltip>
      </HStack>
    </VStack>
  );
};

export default GenericSeriesForm;
