import React, { useMemo, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { Box, Text, Select, HStack, Wrap, WrapItem } from "@chakra-ui/react";

// Фабрика вместо прямого импорта react-plotly.js: обходит несовместимость
// react-plotly.js@4 + plotly.js@3 с webpack-5 (fully-specified ESM).
const Plot = createPlotlyComponent(Plotly);

// Палитра под тёмную тему приложения.
const COLORS = ["#FF0032", "#FFBF00", "#00B5D8", "#48BB78", "#9F7AEA"];

/**
 * Визуализация результатов инференса.
 * Контракт: results = { item_id: { period: { rmse, r2, pred: number[] } } }
 * (см. timecast.serialize_inference_results).
 */
const ForecastChart = ({ results }) => {
  const items = useMemo(() => Object.keys(results || {}), [results]);
  const [selected, setSelected] = useState("");

  if (!items.length) return null;
  const itemId = items.includes(selected) ? selected : items[0];
  const periods = results[itemId] || {};

  const traces = Object.entries(periods)
    .filter(([, v]) => Array.isArray(v.pred) && v.pred.length > 0)
    .map(([period, v], i) => ({
      x: v.pred.map((_, idx) => idx + 1),
      y: v.pred,
      type: "scatter",
      mode: "lines+markers",
      name: `${period}${v.rmse != null ? ` (rmse ${Number(v.rmse).toFixed(2)})` : ""}`,
      line: { color: COLORS[i % COLORS.length], width: 2 },
    }));

  return (
    <Box border="2px solid #FF0032" borderRadius="10px" p={5} bg="#1A1A1A" mt={5} w="100%">
      <HStack mb={3} justify="space-between" align="center">
        <Text fontSize="22px" fontWeight="bold" color="#FFFFFF">
          Прогноз — {itemId}
        </Text>
        <Select
          w="280px"
          value={itemId}
          onChange={(e) => setSelected(e.target.value)}
          bg="#2D2D2D"
          color="#FFFFFF"
          borderColor="#FF0032"
        >
          {items.map((id) => (
            <option key={id} value={id} style={{ color: "#000" }}>
              {id}
            </option>
          ))}
        </Select>
      </HStack>

      {/* Сводка метрик по периодам */}
      <Wrap spacing={4} mb={3}>
        {Object.entries(periods).map(([period, v]) => (
          <WrapItem key={period}>
            <Box bg="#2D2D2D" borderRadius="8px" px={3} py={2}>
              <Text color="#FFBF00" fontWeight="bold">
                {period}
              </Text>
              <Text color="#FFFFFF" fontSize="14px">
                rmse: {v.rmse != null ? Number(v.rmse).toFixed(3) : "—"}
                {"  "}r²: {v.r2 != null ? Number(v.r2).toFixed(3) : "—"}
              </Text>
            </Box>
          </WrapItem>
        ))}
      </Wrap>

      {traces.length ? (
        <Plot
          data={traces}
          layout={{
            autosize: true,
            height: 440,
            paper_bgcolor: "#1A1A1A",
            plot_bgcolor: "#2D2D2D",
            font: { color: "#FFFFFF" },
            margin: { l: 55, r: 20, t: 20, b: 45 },
            xaxis: { title: "Шаг прогноза", gridcolor: "#444" },
            yaxis: { title: "Прогноз спроса", gridcolor: "#444" },
            legend: { orientation: "h" },
          }}
          style={{ width: "100%" }}
          useResizeHandler
          config={{ displayModeBar: false, responsive: true }}
        />
      ) : (
        <Text color="#AAA">Нет числовых предсказаний для отображения.</Text>
      )}
    </Box>
  );
};

export default ForecastChart;
