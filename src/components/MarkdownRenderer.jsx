import React from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import { Box } from "@chakra-ui/react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

const CODE_STYLE = {
  ...vscDarkPlus,
  'pre[class*="language-"]': {
    ...vscDarkPlus['pre[class*="language-"]'],
    background: "#0D1017",
    borderRadius: "10px",
    border: "1px solid #2A2E36",
    fontSize: "13px",
    lineHeight: "1.6",
  },
  'code[class*="language-"]': {
    ...vscDarkPlus['code[class*="language-"]'],
    background: "transparent",
    fontSize: "13px",
  },
};

const MarkdownRenderer = ({ markdownText }) => (
  <Box w="100%">
    <ReactMarkdown
      rehypePlugins={[rehypeRaw]}
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          return !inline && match ? (
            <SyntaxHighlighter
              style={CODE_STYLE}
              language={match[1]}
              PreTag="div"
              customStyle={{ margin: "12px 0" }}
              {...props}
            >
              {String(children).replace(/\n$/, "")}
            </SyntaxHighlighter>
          ) : (
            <Box
              as="code"
              bg="#0D1017"
              color="#FF8888"
              px="5px"
              py="1px"
              borderRadius="4px"
              fontSize="13px"
              fontFamily="monospace"
              {...props}
            >
              {children}
            </Box>
          );
        },
      }}
    >
      {markdownText}
    </ReactMarkdown>
  </Box>
);

export default MarkdownRenderer;
