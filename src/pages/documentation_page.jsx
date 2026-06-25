import React, { useState, useEffect } from "react";
import { Flex, Box, Text, Spinner, HStack, Icon } from "@chakra-ui/react";
import { InfoIcon } from "@chakra-ui/icons";
import MarkdownRenderer from "../components/MarkdownRenderer";

const DocumentationPage = () => {
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${process.env.PUBLIC_URL}/docs/api.md`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((text) => setContent(text))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <Flex direction="column" align="center" w="100%" px={[4, 6, 10]} pt={8} pb={14} flexGrow={1}>
      <Box w="100%" maxW="900px">
        {/* Page header */}
        <HStack mb={6} spacing={3}>
          <Icon as={InfoIcon} color="#FF0032" boxSize="18px" />
          <Text color="#FFFFFF" fontWeight="700" fontSize="22px">
            API Reference
          </Text>
          <Text color="#444" fontSize="13px" pt="3px">
            TimeCast
          </Text>
        </HStack>

        {loading && (
          <Flex justify="center" pt={10}>
            <Spinner color="#FF0032" size="lg" thickness="3px" />
          </Flex>
        )}

        {error && (
          <Box
            bg="#1A0A0A"
            border="1px solid #FF003244"
            borderRadius="10px"
            p={5}
            color="#FF6666"
            fontSize="14px"
          >
            Ошибка загрузки документации: {error}
          </Box>
        )}

        {!loading && !error && (
          <Box
            bg="#141820"
            border="1px solid #2A2E36"
            borderRadius="14px"
            p={[5, 8]}
            sx={{
              // Markdown typography overrides for dark theme
              "h2": { color: "#FFFFFF", fontSize: "20px", fontWeight: 700, mt: "32px", mb: "12px",
                       borderBottom: "1px solid #2A2E36", pb: "8px" },
              "h3": { color: "#CCCCCC", fontSize: "16px", fontWeight: 600, mt: "24px", mb: "8px" },
              "p": { color: "#AAAAAA", fontSize: "14px", lineHeight: "1.75", mb: "12px" },
              "code": { bg: "#0D1017", color: "#FF8888", px: "4px", py: "1px",
                        borderRadius: "4px", fontSize: "13px", fontFamily: "monospace" },
              "pre": { bg: "#0D1017", border: "1px solid #2A2E36", borderRadius: "10px",
                       p: "16px", overflowX: "auto", mb: "16px" },
              "pre code": { bg: "transparent", color: "#E8E8E8", p: 0 },
              "table": { w: "100%", mb: "16px", fontSize: "13px" },
              "th": { color: "#FFBF00", fontWeight: 600, textAlign: "left",
                      borderBottom: "1px solid #2A2E36", pb: "8px", pr: "16px" },
              "td": { color: "#999", borderBottom: "1px solid #1A1D21",
                      py: "6px", pr: "16px", verticalAlign: "top" },
              "hr": { borderColor: "#2A2E36", my: "24px" },
              "ul, ol": { pl: "20px", mb: "12px" },
              "li": { color: "#AAAAAA", fontSize: "14px", mb: "4px" },
              "a": { color: "#FFBF00", _hover: { textDecoration: "underline" } },
            }}
          >
            <MarkdownRenderer markdownText={content} />
          </Box>
        )}
      </Box>
    </Flex>
  );
};

export default DocumentationPage;
