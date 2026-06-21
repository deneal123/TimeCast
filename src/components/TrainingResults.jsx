import React, { useMemo, useState } from "react";
import {
  Box,
  Text,
  Select,
  HStack,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
} from "@chakra-ui/react";

/**
 * Результаты обучения (classic/neiro graduate).
 * Контракт: results = { item_id: { period: { best_model, best_rmse, best_r2, best_param } } }
 * (см. timecast.serialize_training_results).
 */
const TrainingResults = ({ results }) => {
  const items = useMemo(() => Object.keys(results || {}), [results]);
  const [selected, setSelected] = useState("");

  if (!items.length) return null;
  const itemId = items.includes(selected) ? selected : items[0];
  const periods = results[itemId] || {};

  return (
    <Box border="2px solid #FF0032" borderRadius="10px" p={5} bg="#1A1A1A" mt={5} w="100%">
      <HStack mb={3} justify="space-between" align="center">
        <Text fontSize="22px" fontWeight="bold" color="#FFFFFF">
          Обучение — лучшие модели — {itemId}
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

      <TableContainer>
        <Table variant="simple" size="sm">
          <Thead>
            <Tr>
              <Th color="#FFBF00">Период</Th>
              <Th color="#FFBF00">Лучшая модель</Th>
              <Th color="#FFBF00" isNumeric>
                RMSE
              </Th>
              <Th color="#FFBF00" isNumeric>
                R²
              </Th>
            </Tr>
          </Thead>
          <Tbody>
            {Object.entries(periods).map(([period, v]) => (
              <Tr key={period}>
                <Td color="#FFFFFF">{period}</Td>
                <Td color="#FFFFFF">{v.best_model ?? "—"}</Td>
                <Td color="#FFFFFF" isNumeric>
                  {v.best_rmse != null ? Number(v.best_rmse).toFixed(3) : "—"}
                </Td>
                <Td color="#FFFFFF" isNumeric>
                  {v.best_r2 != null ? Number(v.best_r2).toFixed(3) : "—"}
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
