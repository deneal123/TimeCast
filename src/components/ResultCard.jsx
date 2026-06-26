import React from "react";
import { Box, HStack, Text, Badge } from "@chakra-ui/react";

/**
 * Shared dark-themed result panel used by chart and task components.
 *
 * Props:
 *   title        string     header left — main label
 *   badge        string     header left — small badge text (optional)
 *   badgeScheme  string     Chakra colorScheme for the badge (default "gray")
 *   headerRight  node       header right slot — typically an ItemPicker
 *   children     node       card body
 *   ...rest                 forwarded to the outer Box (e.g. mt={5})
 */
const ResultCard = ({
  title,
  badge,
  badgeScheme = "gray",
  headerRight,
  children,
  ...rest
}) => (
  <Box
    bg="#141820"
    border="1px solid #2A2E36"
    borderRadius="14px"
    overflow="hidden"
    w="100%"
    {...rest}
  >
    <HStack
      px={5}
      py={3}
      bg="#0F1218"
      borderBottom="1px solid #2A2E36"
      justify="space-between"
      flexWrap="wrap"
      gap={2}
    >
      <HStack spacing={3}>
        <Text color="#FFFFFF" fontWeight="600" fontSize="15px">
          {title}
        </Text>
        {badge && (
          <Badge
            colorScheme={badgeScheme}
            fontSize="11px"
            px={2}
            borderRadius="6px"
          >
            {badge}
          </Badge>
        )}
      </HStack>
      {headerRight}
    </HStack>
    {children}
  </Box>
);

export default ResultCard;
