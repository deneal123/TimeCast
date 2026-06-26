import React, { useMemo, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import {
  Box,
  Text,
  Select,
  HStack,
  Badge,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
  Divider,
} from "@chakra-ui/react";
import { PLOT_LAYOUT_BASE, PLOT_CONFIG } from "../utils/plotConfig";

const Plot = createPlotlyComponent(Plotly);

const SELECT_STYLE = {
  bg: "#0D1017",
  color: "#E8E8E8",
  border: "1px solid #2A2E36",
  borderRadius: "8px",
  size: "sm",
  _hover: { borderColor: "#444" },
  sx: { option: { background: "#1A1D21" } },
};

const MODEL_HEX = {
  AUTOARIMA: "#4A90D9",
  AUTOREG:   "#00B5D8",
  AUTOETS:   "#48BB78",
  PROPHET:   "#ED8936",
  TBATS:     "#9F7AEA",
  IFFT:      "#FFBF00",
  IF:        "#FFBF00",
};

const MODEL_COLOR = {
  AUTOARIMA: "blue", AUTOREG: "cyan", AUTOETS: "green",
  PROPHET: "orange", TBATS: "purple", IFFT: "yellow", IF: "yellow",
};

const modelColor = (name) => {
  if (!name) return "gray";
  const k = Object.keys(MODEL_COLOR).find((k) => name.toUpperCase().includes(k));
  return k ? MODEL_COLOR[k] : "gray";
};

const modelHex = (name) => {
  if (!name) return "#555";
  const k = Object.keys(MODEL_HEX).find((k) => name.toUpperCase().includes(k));
  return k ? MODEL_HEX[k] : "#888";
};

const r2Color = (val) => {
  if (val == null) return "#666";
  if (val >= 0.8) return "#48BB78";
  if (val >= 0.5) return "#FFBF00";
  return "#FF8888";
};

const TrainingResults = ({ results }) => {
  const items = useMemo(() => Object.keys(results || {}), [results]);
  const [selected, setSelected] = useState("");

  if (!items.length) return null;
  const itemId = items.includes(selected) ? selected : items[0];
  const periods = results[itemId] || {};
  const rows = Object.entries(periods);

  // R² bar chart
  const periodList = rows.map(([p]) => p);
  const r2Values   = rows.map(([, v]) => Math.max(0, v?.best_r2 ?? 0));
  const modelNames = rows.map(([, v]) => v?.best_model ?? "—");
  const hasChart   = r2Values.some((v) => v > 0);

  const barTrace = {
    x: r2Values,
    y: periodList,
    type: "bar",
    orientation: "h",
    marker: { color: modelNames.map(modelHex), opacity: 0.9 },
    text: r2Values.map((v) => v.toFixed(3)),
    textposition: "inside",
    insidetextanchor: "end",
    textfont: { color: "#FFFFFF", size: 11 },
    customdata: modelNames,
    hovertemplate: "<b>%{y}</b><br>R² %{x:.4f}<br>%{customdata}<extra></extra>",
  };

  const barLayout = {
    ...PLOT_LAYOUT_BASE,
    height: Math.max(140, periodList.length * 52 + 70),
    xaxis: {
      ...PLOT_LAYOUT_BASE.xaxis,
      range: [0, 1.05],
      title: { text: "R²", font: { color: "#555", size: 12 } },
      tickformat: ".2f",
    },
    yaxis: {
      ...PLOT_LAYOUT_BASE.yaxis,
      title: null,
      automargin: true,
    },
    margin: { l: 70, r: 30, t: 10, b: 50 },
    showlegend: false,
    bargap: 0.3,
  };

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
        px={5} py={3}
        bg="#0F1218"
        borderBottom="1px solid #2A2E36"
        justify="space-between"
      >
        <HStack spacing={3}>
          <Text color="#FFFFFF" fontWeight="600" fontSize="15px">
            Результаты обучения
          </Text>
          <Badge colorScheme="yellow" fontSize="11px" px={2} borderRadius="6px">
            training
          </Badge>
        </HStack>
        <Select
          {...SELECT_STYLE}
          w="220px"
          value={itemId}
          onChange={(e) => setSelected(e.target.value)}
        >
          {items.map((id) => (
            <option key={id} value={id}>{id}</option>
          ))}
        </Select>
      </HStack>

      {/* R² bar chart */}
      {hasChart && (
        <>
          <Box px={5} pt={4} pb={2}>
            <Text color="#555" fontSize="11px" fontWeight="600" letterSpacing="0.08em" mb={2}>
              R² ПО ПЕРИОДАМ
            </Text>
            <Plot
              data={[barTrace]}
              layout={barLayout}
              style={{ width: "100%" }}
              useResizeHandler
              config={PLOT_CONFIG}
            />
          </Box>
          <Divider borderColor="#2A2E36" />
        </>
      )}

      {/* Table */}
      <TableContainer px={5} py={4}>
        <Table variant="unstyled" size="sm">
          <Thead>
            <Tr>
              {["Период", "Лучшая модель", "RMSE", "R²"].map((h) => (
                <Th
                  key={h}
                  color="#555"
                  fontSize="11px"
                  fontWeight="600"
                  letterSpacing="0.08em"
                  textTransform="uppercase"
                  borderBottom="1px solid #2A2E36"
                  pb={3}
                  isNumeric={h === "RMSE" || h === "R²"}
                >
                  {h}
                </Th>
              ))}
            </Tr>
          </Thead>
          <Tbody>
            {rows.map(([period, v], idx) => (
              <Tr
                key={period}
                bg={idx % 2 === 0 ? "transparent" : "#0D101733"}
                _hover={{ bg: "#0D1017" }}
                transition="background 0.15s"
              >
                <Td color="#AAAAAA" borderBottom="1px solid #1A1D21" py={3}
                  fontFamily="monospace" fontSize="13px">
                  {period}
                </Td>
                <Td borderBottom="1px solid #1A1D21" py={3}>
                  {v.best_model ? (
                    <Badge
                      colorScheme={modelColor(v.best_model)}
                      fontSize="12px" px={2} borderRadius="6px"
                    >
                      {v.best_model}
                    </Badge>
                  ) : (
                    <Text color="#555">—</Text>
                  )}
                </Td>
                <Td isNumeric color="#CCCCCC" borderBottom="1px solid #1A1D21"
                  py={3} fontFamily="monospace" fontSize="13px">
                  {v.best_rmse != null ? Number(v.best_rmse).toFixed(4) : "—"}
                </Td>
                <Td isNumeric borderBottom="1px solid #1A1D21" py={3}
                  fontFamily="monospace" fontSize="13px"
                  color={r2Color(v.best_r2)} fontWeight="600">
                  {v.best_r2 != null ? Number(v.best_r2).toFixed(4) : "—"}
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default TrainingResults;
