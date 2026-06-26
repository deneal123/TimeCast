import React, { useMemo, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { Box, Text, Select, HStack, Badge, Flex, SimpleGrid } from "@chakra-ui/react";
import { PLOT_LAYOUT_BASE, PLOT_CONFIG } from "../utils/plotConfig";

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

const SELECT_STYLE = {
  bg: "#0D1017",
  color: "#E8E8E8",
  border: "1px solid #2A2E36",
  borderRadius: "8px",
  size: "sm",
  _hover: { borderColor: "#444" },
  sx: { option: { background: "#1A1D21" } },
};

const StatCard = ({ label, color, values }) => {
  if (!Array.isArray(values) || !values.length) return null;
  const nums = values.filter((v) => typeof v === "number");
  const min = Math.min(...nums).toFixed(2);
  const max = Math.max(...nums).toFixed(2);
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
      <Text color={color} fontWeight="600" fontSize="13px" mb={1}>
        {label}
      </Text>
      <HStack spacing={3}>
        {[["min", min], ["mean", mean], ["max", max]].map(([k, v]) => (
          <Box key={k}>
            <Text color="#444" fontSize="10px" fontWeight="600" letterSpacing="0.06em">
              {k.toUpperCase()}
            </Text>
            <Text color="#AAAAAA" fontSize="12px" fontFamily="monospace">
              {v}
            </Text>
          </Box>
        ))}
      </HStack>
    </Box>
  );
};

const DecompositionChart = ({ results }) => {
  const items = useMemo(() => Object.keys(results || {}), [results]);
  const [selItem, setSelItem] = useState("");
  const [selPeriod, setSelPeriod] = useState("");

  if (!items.length) return null;
  const itemId = items.includes(selItem) ? selItem : items[0];
  const periods = Object.keys(results[itemId] || {});
  const period = periods.includes(selPeriod) ? selPeriod : periods[0];
  const comp = (results[itemId] || {})[period] || {};

  const xOf = (arr) => arr.map((_, i) => i + 1);
  const traces = COMPONENTS
    .filter(({ key }) => Array.isArray(comp[key]) && comp[key].length > 0)
    .map(({ key, color, label }) => ({
      x: xOf(comp[key]),
      y: comp[key],
      type: "scatter",
      mode: "lines",
      name: label,
      line: { color, width: 2 },
    }));

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
            Сезонная декомпозиция
          </Text>
          <Badge colorScheme="purple" fontSize="11px" px={2} borderRadius="6px">
            decompose
          </Badge>
        </HStack>
        <HStack spacing={3}>
          <Select
            {...SELECT_STYLE}
            w="200px"
            value={itemId}
            onChange={(e) => setSelItem(e.target.value)}
          >
            {items.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </Select>
          {periods.length > 1 && (
            <Select
              {...SELECT_STYLE}
              w="130px"
              value={period}
              onChange={(e) => setSelPeriod(e.target.value)}
            >
              {periods.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </Select>
          )}
        </HStack>
      </HStack>

      {/* Stat cards */}
      <SimpleGrid columns={[1, 3]} spacing={3} px={5} pt={4} pb={2}>
        {COMPONENTS.map(({ key, color, label }) => (
          <StatCard key={key} label={label} color={color} values={comp[key]} />
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
            config={PLOT_CONFIG}
          />
        ) : (
          <Flex align="center" justify="center" h="200px">
            <Text color="#444" fontSize="14px">
              Нет данных декомпозиции
            </Text>
          </Flex>
        )}
      </Box>
    </Box>
  );
};

export default DecompositionChart;
