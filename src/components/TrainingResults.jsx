import React, { useMemo, useState } from "react";
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
} from "@chakra-ui/react";

const SELECT_STYLE = {
  bg: "#0D1017",
  color: "#E8E8E8",
  border: "1px solid #2A2E36",
  borderRadius: "8px",
  size: "sm",
  _hover: { borderColor: "#444" },
  sx: { option: { background: "#1A1D21" } },
};

// Цвет бейджа по типу модели
const MODEL_COLOR = {
  AUTOARIMA: "blue",
  AUTOREG: "cyan",
  AUTOETS: "green",
  PROPHET: "orange",
  TBATS: "purple",
  IFFT: "yellow",
  IF: "yellow",
};

const modelColor = (name) => {
  if (!name) return "gray";
  const key = Object.keys(MODEL_COLOR).find((k) => name.toUpperCase().includes(k));
  return key ? MODEL_COLOR[key] : "gray";
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
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </Select>
      </HStack>

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
                <Td
                  color="#AAAAAA"
                  borderBottom="1px solid #1A1D21"
                  py={3}
                  fontFamily="monospace"
                  fontSize="13px"
                >
                  {period}
                </Td>
                <Td borderBottom="1px solid #1A1D21" py={3}>
                  {v.best_model ? (
                    <Badge
                      colorScheme={modelColor(v.best_model)}
                      fontSize="12px"
                      px={2}
                      borderRadius="6px"
                    >
                      {v.best_model}
                    </Badge>
                  ) : (
                    <Text color="#555">—</Text>
                  )}
                </Td>
                <Td
                  isNumeric
                  color="#CCCCCC"
                  borderBottom="1px solid #1A1D21"
                  py={3}
                  fontFamily="monospace"
                  fontSize="13px"
                >
                  {v.best_rmse != null ? Number(v.best_rmse).toFixed(4) : "—"}
                </Td>
                <Td
                  isNumeric
                  borderBottom="1px solid #1A1D21"
                  py={3}
                  fontFamily="monospace"
                  fontSize="13px"
                  color={r2Color(v.best_r2)}
                  fontWeight="600"
                >
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
