import React from "react";
import { HStack, Select, Button, Tooltip } from "@chakra-ui/react";

const SELECT_BASE = {
  bg: "#0D1017",
  color: "#E8E8E8",
  border: "1px solid #2A2E36",
  borderRadius: "8px",
  size: "sm",
  _hover: { borderColor: "#444" },
  sx: { option: { background: "#1A1D21" } },
};

/**
 * Unified item picker:
 * - ≤5 items  → tab-style buttons
 * - 6+ items  → <Select> dropdown
 * - ≤1 items  → null (no picker needed)
 *
 * Props:
 *   items        string[]   list of IDs
 *   selected     string     currently active ID
 *   onSelect     fn(id)     callback on selection
 *   accentColor  string     active-state color (default #FF0032)
 *   selectWidth  string     dropdown width (default "200px")
 */
const ItemPicker = ({
  items,
  selected,
  onSelect,
  accentColor = "#FF0032",
  selectWidth = "200px",
}) => {
  if (!items || items.length <= 1) return null;

  if (items.length <= 5) {
    return (
      <HStack spacing={1} flexWrap="wrap" justify="flex-end">
        {items.map((id) => {
          const active = id === selected;
          return (
            <Tooltip
              key={id}
              label={id}
              placement="top"
              hasArrow
              isDisabled={id.length <= 14}
            >
              <Button
                size="xs"
                variant="ghost"
                bg={active ? `${accentColor}1A` : "transparent"}
                color={active ? accentColor : "#555"}
                border="1px solid"
                borderColor={active ? `${accentColor}44` : "transparent"}
                _hover={{ color: "#CCCCCC", borderColor: "#2A2E36" }}
                borderRadius="6px"
                onClick={() => onSelect(id)}
                px={3}
                h="26px"
                fontSize="12px"
                fontWeight={active ? "600" : "400"}
                transition="all 0.15s"
                maxW="120px"
                isTruncated
              >
                {id}
              </Button>
            </Tooltip>
          );
        })}
      </HStack>
    );
  }

  return (
    <Select
      {...SELECT_BASE}
      w={selectWidth}
      value={selected}
      onChange={(e) => onSelect(e.target.value)}
    >
      {items.map((id) => (
        <option key={id} value={id}>
          {id}
        </option>
      ))}
    </Select>
  );
};

export default ItemPicker;
