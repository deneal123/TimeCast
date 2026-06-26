import React, { useMemo, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { Box, Text, HStack, SimpleGrid, Flex } from "@chakra-ui/react";
import { PLOT_LAYOUT_BASE, PLOT_CONFIG } from "../utils/plotConfig";
import ItemPicker from "./ItemPicker";
import ResultCard from "./ResultCard";

const Plot = createPlotlyComponent(Plotly);

const COMPONENTS = [
  { key: "trend",    color: "#FF0032", label: "Тренд" },
  { key: "seasonal", color: "#FFBF00", label: "Сезонность" },
  { key: "resid",    color: "#00B5D8", label: "Остаток" },
];

const PLOT_LAYOUT = {
  ...PLOT_LAYOUT_BASE,
  height: 380,
  xaxis: {
    ...PLOT_LAYOUT_BASE.xaxis,
    title: { text: "Шаг", font: { color: "#555", size: 12 } },
  },
};

const StatCard = ({ label, color, values }) => {
  if (!Array.isArray(values) || !values.length) return null;
  const nums = values.filter((v) => typeof v === "number");
  const min  = Math.min(...nums).toFixed(2);
  const max  = Math.max(...nums).toFixed(2);
  const mean = (nums.reduce((a, b) => a + b, 0) / nums.length).toFixed(2);

  return (
    <Box
      bg="#0D1017"
      border="1px solid #2A2E36"
      borderLeft={`3px solid ${color}`}
      borderRadius="10px"
      px={4}
      py={3}
    >
      <Text color={color} fontWeight="600" fontSize="13px" mb={1}>{label}</Text>
      <HStack spacing={3}>
        {[["min", min], ["mean", mean], ["max", max]].map(([k, v]) => (
          <Box key={k}>
            <Text color="#444" fontSize="10px" fontWeight="600" letterSpacing="0.06em">
              {k.toUpperCase()}
            </Text>
            <Text color="#AAAAAA" fontSize="12px" fontFamily="monospace">{v}</Text>
          </Box>
        ))}
      </HStack>
    </Box>
  );
};

const DecompositionChart = ({ results }) => {
  const items = useMemo(() => Object.keys(results || {}), [results]);
  const [selItem,   setSelItem]   = useState("");
  const [selPeriod, setSelPeriod] = useState("");

  if (!items.length) return null;
  const itemId  = items.includes(selItem)    ? selItem   : items[0];
  const periods = Object.keys(results[itemId] || {});
  const period  = periods.includes(selPeriod) ? selPeriod : periods[0];
  const comp    = (results[itemId] || {})[period] || {};

  const xOf   = (arr) => arr.map((_, i) => i + 1);
  const traces = COMPONENTS
    .filter(({ key }) => Array.isArray(comp[key]) && comp[key].length > 0)
    .map(({ key, color, label }) => ({
      x: xOf(comp[key]), y: comp[key],
      type: "scatter", mode: "lines", name: label,
      line: { color, width: 2 },
    }));

  return (
    <ResultCard
      title="Сезонная декомпозиция"
      badge="decompose"
      badgeScheme="purple"
      headerRight={
        <HStack spacing={2} flexWrap="wrap" justify="flex-end">
          <ItemPicker
            items={items}
            selected={itemId}
            onSelect={(v) => { setSelItem(v); setSelPeriod(""); }}
            accentColor="#9F7AEA"
          />
          <ItemPicker
            items={periods}
            selected={period}
            onSelect={setSelPeriod}
            accentColor="#00B5D8"
          />
        </HStack>
      }
      mt={5}
    >
      <SimpleGrid columns={[1, 3]} spacing={3} px={5} pt={4} pb={2}>
        {COMPONENTS.map(({ key, color, label }) => (
          <StatCard key={key} label={label} color={color} values={comp[key]} />
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
            <Text color="#444" fontSize="14px">Нет данных декомпозиции</Text>
          </Flex>
        )}
      </Box>
    </ResultCard>
  );
};

export default DecompositionChart;
