import React, { useState, useEffect } from "react";
import { Flex, VStack, Text, Spinner } from "@chakra-ui/react";
import useWindowDimensions from "../hooks/window_dimensions";
import MarkdownRenderer from "../components/MarkdownRenderer";

const DocumentationPage = () => {
  const { width } = useWindowDimensions();
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
    <Flex
      direction="column"
      bg="transparent"
      padding={25}
      flexGrow={1}
      align="center"
      justify="flex-start"
      width={width}
      height="100%"
      overflowX="hidden"
      overflowY="auto"
      paddingTop="150px"
    >
      <VStack spacing={4} align="stretch" width="100%" maxW="1200px">
        {loading && <Spinner color="#FF0032" size="lg" />}
        {error && (
          <Text color="#FF0032">Ошибка загрузки документации: {error}</Text>
        )}
        {!loading && !error && (
          <Flex
            direction="column"
            width="100%"
            overflowY="auto"
            padding="16px"
            bg="gray.100"
            borderRadius="8px"
          >
            <MarkdownRenderer markdownText={content} />
          </Flex>
        )}
      </VStack>
    </Flex>
  );
};

export default DocumentationPage;
