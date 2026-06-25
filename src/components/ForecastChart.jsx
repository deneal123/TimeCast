import React, { useMemo, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import {
  Box,
  Text,
  Select,
  HStack,
  SimpleGrid,
  Badge,
  Flex,
} from "@chakra-ui/react";

const Plot = createPlotlyComponent(Plotly);

const COLORS = ["#FF0032", "#FFBF00", "#00B5D8", "#48BB78", "#9F7AEA"];

const PLOT_LAYOUT = {
  autosize: true,
  height: 400,
  paper_bgcolor: "#141820",
  plot_bgcolor: "#0D1017",
  font: { color: "#888", size: 12, family: "Inter, sans-serif" },
  margin: { l: 60, r: 20, t: 16, b: 50 },
  xaxis: {
    title: { text: "Шаг прогноза", font: { color: "#555", size: 12 } },
    gridcolor: "#1E2330",
    linecolor: "#2A2E36",
    zerolinecolor: "#2A2E36",
    tickfont: { color: "#666" },
  },
  yaxis: {
    title: { text: "Значение", font: { color: "#555", size: 12 } },
    gridcolor: "#1E2330",
    linecolor: "#2A2E36",
    zerolinecolor: "#2A2E36",
    tickfont: { color: "#666" },
  },
  legend: {
    orientation: "h",
    y: -0.15,
    bgcolor: "transparent",
    borderwidth: 0,
    font: { color: "#888", size: 12 },
  },
  hovermode: "x unified",
  hoverlabel: { bgcolor: "#1A1D21", bordercolor: "#2A2E36", font: { color: "#FFFFFF" } },
};

const SELECT_STYLE = {
  bg: "#0D1017",
  color: "#E8E8E8",
  border: "1px solid #2A2E36",
  borderRadius: "8px",
  size: "sm",
  _hover: { borderColor: "#444" },
  sx: { option: { background: "#1A1D21" } },
};

const r2Color = (val) => {
  if (val == null) return "#666";
  if (val >= 0.8) return "#48BB78";
  if (val >= 0.5) return "#FFBF00";
  return "#FF0032";
};

const MetricCard = ({ period, rmse, r2, color }) => (
  <Box
    bg="#0D1017"
    border="1px solid #2A2E36"
    borderLeft={`3px solid ${color}`}
    borderRadius="10px"
    px={4}
    py={3}
  >
    <HStack justify="space-between" mb={1}>
      <Text color={color} fontWeight="600" fontSize="13px">
        {period}
      </Text>
      <Badge
        bg={`${r2Color(r2)}22`}
        color={r2Color(r2)}
        fontSize="10px"
        borderRadius="4px"
        px={2}
      >
        R² {r2 != null ? Number(r2).toFixed(3) : "—"}
      </Badge>
    </HStack>
    <Text color="#666" fontSize="12px">
      RMSE&nbsp;
      <Text as="span" color="#AAAAAA" fontWeight="600">
        {rmse != null ? Number(rmse).toFixed(3) : "—"}
      </Text>
    </Text>
  </Box>
);

const ForecastChart = ({ results }) => {
  const items = useMemo(() => Object.keys(results || {}), [results]);
  const [selected, setSelected] = useState("");

  if (!items.length) return null;
  const itemId = items.includes(selected) ? selected : items[0];
  const periods = results[itemId] || {};

  const traces = [];
  Object.entries(periods)
    .filter(([, v]) => Array.isArray(v.pred) && v.pred.length > 0)
    .forEach(([period, v], i) => {
      const color = COLORS[i % COLORS.length];
      if (i === 0 && Array.isArray(v.actual) && v.actual.length > 0) {
        traces.push({
          x: v.actual.map((_, idx) => idx + 1),
          y: v.actual,
          type: "scatter",
          mode: "lines",
          name: "факт",
          line: { color: "#555555", width: 1.5, dash: "dot" },
          opacity: 0.7,
        });
      }
      traces.push({
        x: v.pred.map((_, idx) => idx + 1),
        y: v.pred,
        type: "scatter",
        mode: "lines+markers",
        name: period,
        line: { color, width: 2 },
        marker: { color, size: 4 },
      });
    });

  return (
    <Box
      bg="#141820"
      border="1px solid #2A2E36"
      borderRadius="14px"
      overflow="hidden"
      mt={5}
      w="100%"
    >
      {/* Header */}
      <HStack
        px={5}
        py={3}
        bg="#0F1218"
        borderBottom="1px solid #2A2E36"
        justify="space-between"
      >
        <HStack spacing={3}>
          <Text color="#FFFFFF" fontWeight="600" fontSize="15px">
            Прогноз
          </Text>
          <Badge colorScheme="red" fontSize="11px" px={2} borderRadius="6px">
            inference
          </Badge>
        </HStack>
        <Select
          {...SELECT_STYLE}
          w="220px"
          value={itemId}
          onChange={(e) => setSelected(e.target.value)}
        >
          {items.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </Select>
      </HStack>

      {/* Metric cards */}
      <SimpleGrid columns={[2, 3, 4]} spacing={3} px={5} pt={4} pb={2}>
        {Object.entries(periods).map(([period, v], i) => (
          <MetricCard
            key={period}
            period={period}
            rmse={v.rmse}
            r2={v.r2}
            color={COLORS[i % COLORS.length]}
          />
        ))}
      </SimpleGrid>

      {/* Chart */}
      <Box px={3} pb={4}>
        {traces.length ? (
          <Plot
            data={traces}
            layout={PLOT_LAYOUT}
            style={{ width: "100%" }}
            useResizeHandler
            config={{ displayModeBar: false, responsive: true }}
          />
        ) : (
          <Flex align="center" justify="center" h="200px">
            <Text color="#444" fontSize="14px">
              Нет числовых предсказаний для отображения
            </Text>
          </Flex>
        )}
      </Box>
    </Box>
  );
};

export default ForecastChart;
