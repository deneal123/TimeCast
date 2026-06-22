import React, { useMemo, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { Box, Text, Select, HStack } from "@chakra-ui/react";

const Plot = createPlotlyComponent(Plotly);

/**
 * Сезонная декомпозиция (аддитивная): тренд / сезон / остаток.
 * Контракт: results = { item_id: { period: { trend, seasonal, resid } } }
 * (см. timecast.serialize_decomposition_results).
 */
const DecompositionChart = ({ results }) => {
  const items = useMemo(() => Object.keys(results || {}), [results]);
  const [selItem, setSelItem] = useState("");
  const [selPeriod, setSelPeriod] = useState("");

  if (!items.length) return null;
  const itemId = items.includes(selItem) ? selItem : items[0];
  const periods = Object.keys(results[itemId] || {});
  const period = periods.includes(selPeriod) ? selPeriod : periods[0];
  const comp = (results[itemId] || {})[period] || {};

  const xOf = (arr) => (Array.isArray(arr) ? arr.map((_, i) => i + 1) : []);
  const traces = [
    { key: "trend", color: "#FF0032" },
    { key: "seasonal", color: "#FFBF00" },
    { key: "resid", color: "#00B5D8" },
  ]
    .filter(({ key }) => Array.isArray(comp[key]) && comp[key].length > 0)
    .map(({ key, color }) => ({
      x: xOf(comp[key]),
      y: comp[key],
      type: "scatter",
      mode: "lines",
      name: key,
      line: { color, width: 2 },
    }));

  return (
    <Box border="2px solid #FF0032" borderRadius="10px" p={5} bg="#1A1A1A" mt={5} w="100%">
      <HStack mb={3} justify="space-between" align="center" spacing={4}>
        <Text fontSize="22px" fontWeight="bold" color="#FFFFFF">
          Декомпозиция — {itemId}
        </Text>
        <HStack spacing={3}>
          <Select
            w="220px"
            value={itemId}
            onChange={(e) => setSelItem(e.target.value)}
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
          <Select
            w="140px"
            value={period}
            onChange={(e) => setSelPeriod(e.target.value)}
            bg="#2D2D2D"
            color="#FFFFFF"
            borderColor="#FF0032"
          >
            {periods.map((p) => (
              <option key={p} value={p} style={{ color: "#000" }}>
                {p}
              </option>
            ))}
          </Select>
        </HStack>
      </HStack>

      {traces.length ? (
        <Plot
          data={traces}
          layout={{
            autosize: true,
            height: 460,
            paper_bgcolor: "#1A1A1A",
            plot_bgcolor: "#2D2D2D",
            font: { color: "#FFFFFF" },
            margin: { l: 55, r: 20, t: 20, b: 45 },
            xaxis: { title: "Шаг", gridcolor: "#444" },
            yaxis: { title: "Значение", gridcolor: "#444" },
            legend: { orientation: "h" },
          }}
          style={{ width: "100%" }}
          useResizeHandler
          config={{ displayModeBar: false, responsive: true }}
        />
      ) : (
        <Text color="#AAA">Нет данных декомпозиции для отображения.</Text>
      )}
    </Box>
  );
};

export default DecompositionChart;
