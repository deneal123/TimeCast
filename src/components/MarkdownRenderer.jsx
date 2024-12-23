import React from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw"; // Плагин для обработки HTML в Markdown
import { Box } from "@chakra-ui/react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { solarizedlight } from "react-syntax-highlighter/dist/esm/styles/prism";

const MarkdownRenderer = ({ markdownText }) => {
    return (
        <Box width="100%" maxW="1200px" p={6}>
            <ReactMarkdown
                children={markdownText}
                rehypePlugins={[rehypeRaw]} // Подключаем плагин для поддержки HTML
                components={{
                    code({ node, inline, className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || '');
                        return !inline && match ? (
                            <SyntaxHighlighter
                                style={solarizedlight}
                                language={match[1]}
                                PreTag="div"
                                {...props}
                            >
                                {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                        ) : (
                            <code className={className} {...props}>
                                {children}
                            </code>
                        );
                    }
                }}
            />
        </Box>
    );
};

export default MarkdownRenderer;