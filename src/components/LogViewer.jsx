import React from "react";
import { Box, Text } from "@chakra-ui/react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

const JSON_STYLE = {
  ...vscDarkPlus,
  'pre[class*="language-"]': {
    ...vscDarkPlus['pre[class*="language-"]'],
    background: "transparent",
    margin: 0,
    padding: 0,
    fontSize: "12px",
    lineHeight: "1.65",
  },
  'code[class*="language-"]': {
    ...vscDarkPlus['code[class*="language-"]'],
    background: "transparent",
    fontSize: "12px",
  },
};

const SCROLL_SX = {
  "&::-webkit-scrollbar": { w: "6px" },
  "&::-webkit-scrollbar-track": { bg: "#0D1017" },
  "&::-webkit-scrollbar-thumb": { bg: "#2A2E36", borderRadius: "3px" },
};

const lineColor = (line) => {
  if (/\b(ERROR|FAIL(ED)?|Exception|Traceback)\b/i.test(line)) return "#FF8888";
  if (/\b(WARN(ING)?)\b/i.test(line)) return "#FFBF00";
  if (/\b(INFO)\b/i.test(line)) return "#88BBFF";
  if (/\b(DEBUG)\b/i.test(line)) return "#888888";
  if (/\b(SUCCESS|DONE|COMPLETE|OK)\b|✓/i.test(line)) return "#48BB78";
  return "#AAAAAA";
};

const LogViewer = React.forwardRef(({ value, height = "480px" }, ref) => {
  const trimmed = (value || "").trim();
  const isJson = trimmed.startsWith("{") || trimmed.startsWith("[");

  return (
    <Box
      ref={ref}
      h={height}
      overflowY="auto"
      bg="#0D1017"
      border="1px solid #2A2E36"
      borderRadius="10px"
      p={3}
      sx={SCROLL_SX}
    >
      {isJson ? (
        <SyntaxHighlighter
          style={JSON_STYLE}
          language="json"
          customStyle={{ background: "transparent", margin: 0, padding: 0 }}
        >
          {trimmed}
        </SyntaxHighlighter>
      ) : trimmed ? (
        trimmed.split("\n").map((line, i) => (
          <Text
            key={i}
            color={lineColor(line)}
            fontSize="12px"
            fontFamily="monospace"
            lineHeight="1.65"
            whiteSpace="pre-wrap"
            wordBreak="break-all"
          >
            {line || " "}
          </Text>
        ))
      ) : (
        <Text color="#2A2E36" fontSize="12px" fontFamily="monospace">
          Ожидание ответа…
        </Text>
      )}
    </Box>
  );
});

LogViewer.displayName = "LogViewer";
export default LogViewer;
