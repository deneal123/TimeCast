import React, { useMemo, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { Box, Text, Badge, HStack, SimpleGrid, Flex } from "@chakra-ui/react";
import { PLOT_LAYOUT_BASE, PLOT_CONFIG } from "../utils/plotConfig";
import ItemPicker from "./ItemPicker";
import ResultCard from "./ResultCard";

const Plot = createPlotlyComponent(Plotly);

const COLORS = ["#FF0032", "#FFBF00", "#00B5D8", "#48BB78", "#9F7AEA"];

const PLOT_LAYOUT = {
  ...PLOT_LAYOUT_BASE,
  height: 400,
  xaxis: {
    ...PLOT_LAYOUT_BASE.xaxis,
    title: { text: "Шаг прогноза", font: { color: "#555", size: 12 } },
  },
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
      <Text color={color} fontWeight="600" fontSize="13px">{period}</Text>
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
      RMSE{" "}
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
  const itemId  = items.includes(selected) ? selected : items[0];
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
          type: "scatter", mode: "lines", name: "факт",
          line: { color: "#555555", width: 1.5, dash: "dot" },
          opacity: 0.7,
        });
      }
      traces.push({
        x: v.pred.map((_, idx) => idx + 1),
        y: v.pred,
        type: "scatter", mode: "lines+markers", name: period,
        line: { color, width: 2 },
        marker: { color, size: 4 },
      });
    });

  return (
    <ResultCard
      title="Прогноз"
      badge="inference"
      badgeScheme="red"
      headerRight={
        <ItemPicker
          items={items}
          selected={itemId}
          onSelect={setSelected}
          accentColor="#FF0032"
        />
      }
      mt={5}
    >
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

      <Box px={3} pb={4}>
        {traces.length ? (
          <Plot
            data={traces}
            layout={PLOT_LAYOUT}
            style={{ width: "100%" }}
            useResizeHandler
            config={PLOT_CONFIG}
          />
        ) : (
          <Flex align="center" justify="center" h="200px">
            <Text color="#444" fontSize="14px">
              Нет числовых предсказаний для отображения
            </Text>
          </Flex>
        )}
      </Box>
    </ResultCard>
  );
};

export default React.memo(ForecastChart);
