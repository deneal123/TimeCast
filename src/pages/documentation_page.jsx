import React, { useState, useEffect, useMemo } from "react";
import {
  Flex, Box, Text, HStack, Icon,
  Skeleton, SkeletonText,
} from "@chakra-ui/react";
import { InfoIcon } from "@chakra-ui/icons";
import MarkdownRenderer from "../components/MarkdownRenderer";

// ---- Heading extraction (mirrors headingId() in MarkdownRenderer) ----
const slugify = (text) =>
  text
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^\wЀ-ӿ-]/g, "")
    .replace(/^-+|-+$/g, "");

const extractHeadings = (markdown) => {
  if (!markdown) return [];
  return markdown
    .split("\n")
    .filter((l) => /^#{2,3} /.test(l))
    .map((l) => {
      const m = l.match(/^(#{2,3}) (.+)/);
      const level = m[1].length;
      const text  = m[2].trim();
      return { level, text, id: slugify(text) };
    });
};

// ---- Skeleton ----
const DocSkeleton = () => (
  <Box bg="#141820" border="1px solid #2A2E36" borderRadius="14px" p={[5, 8]}>
    <Skeleton h="22px" w="180px" mb={6} startColor="#1A1D21" endColor="#2A2E36" />
    <SkeletonText noOfLines={4} spacing={4} skeletonHeight="13px" startColor="#1A1D21" endColor="#2A2E36" mb={8} />
    <Skeleton h="18px" w="220px" mb={4} startColor="#1A1D21" endColor="#2A2E36" />
    <SkeletonText noOfLines={5} spacing={4} skeletonHeight="13px" startColor="#1A1D21" endColor="#2A2E36" mb={8} />
    <Skeleton h="18px" w="160px" mb={4} startColor="#1A1D21" endColor="#2A2E36" />
    <Skeleton h="80px" borderRadius="10px" mb={6} startColor="#1A1D21" endColor="#2A2E36" />
    <SkeletonText noOfLines={6} spacing={4} skeletonHeight="13px" startColor="#1A1D21" endColor="#2A2E36" />
  </Box>
);

// ---- TOC sidebar ----
const TableOfContents = ({ headings, activeId }) => {
  if (!headings.length) return null;

  return (
    <Box
      display={["none", "none", "none", "block"]}
      w="190px"
      flexShrink={0}
      position="sticky"
      top="80px"
      maxH="calc(100vh - 100px)"
      overflowY="auto"
      sx={{
        "&::-webkit-scrollbar": { w: "4px" },
        "&::-webkit-scrollbar-track": { bg: "transparent" },
        "&::-webkit-scrollbar-thumb": { bg: "#2A2E36", borderRadius: "2px" },
      }}
      pr={2}
    >
      <Text
        color="#333"
        fontSize="10px"
        fontWeight="700"
        letterSpacing="0.1em"
        textTransform="uppercase"
        mb={3}
      >
        Содержание
      </Text>

      {headings.map(({ level, text, id }) => (
        <Box
          key={id}
          pl={level === 3 ? 3 : 0}
          py="5px"
          cursor="pointer"
          onClick={() => {
            document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
          }}
        >
          <Text
            fontSize="12px"
            color={activeId === id ? "#FFBF00" : level === 2 ? "#666" : "#444"}
            fontWeight={activeId === id ? "600" : "400"}
            _hover={{ color: "#CCCCCC" }}
            transition="color 0.15s"
            lineHeight="1.4"
            noOfLines={2}
          >
            {text}
          </Text>
        </Box>
      ))}
    </Box>
  );
};

// ---- Main page ----
const DocumentationPage = () => {
  const [content,  setContent]  = useState("");
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState(null);
  const [activeId, setActiveId] = useState("");

  const headings = useMemo(() => extractHeadings(content), [content]);

  useEffect(() => {
    fetch(`${process.env.PUBLIC_URL}/docs/api.md`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
      .then(setContent)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // IntersectionObserver for active TOC item
  useEffect(() => {
    if (!headings.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const first = entries.find((e) => e.isIntersecting);
        if (first) setActiveId(first.target.id);
      },
      { rootMargin: "-72px 0px -65% 0px", threshold: 0 }
    );

    headings.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [headings]);

  const contentSx = {
    h2: { color: "#FFFFFF", fontSize: "20px", fontWeight: 700, mt: "32px", mb: "12px",
          borderBottom: "1px solid #2A2E36", pb: "8px" },
    h3: { color: "#CCCCCC", fontSize: "16px", fontWeight: 600, mt: "24px", mb: "8px" },
    p:  { color: "#AAAAAA", fontSize: "14px", lineHeight: "1.75", mb: "12px" },
    code: { bg: "#0D1017", color: "#FF8888", px: "4px", py: "1px",
            borderRadius: "4px", fontSize: "13px", fontFamily: "monospace" },
    pre: { bg: "#0D1017", border: "1px solid #2A2E36", borderRadius: "10px",
           p: "16px", overflowX: "auto", mb: "16px" },
    "pre code": { bg: "transparent", color: "#E8E8E8", p: 0 },
    table: { w: "100%", mb: "16px", fontSize: "13px" },
    th: { color: "#FFBF00", fontWeight: 600, textAlign: "left",
          borderBottom: "1px solid #2A2E36", pb: "8px", pr: "16px" },
    td: { color: "#999", borderBottom: "1px solid #1A1D21",
          py: "6px", pr: "16px", verticalAlign: "top" },
    hr: { borderColor: "#2A2E36", my: "24px" },
    "ul, ol": { pl: "20px", mb: "12px" },
    li: { color: "#AAAAAA", fontSize: "14px", mb: "4px" },
    a:  { color: "#FFBF00", _hover: { textDecoration: "underline" } },
  };

  return (
    <Flex direction="column" align="center" w="100%" px={[4, 6, 10]} pt={8} pb={14} flexGrow={1}>
      <Box w="100%" maxW="1200px">

        {/* Page header */}
        <HStack mb={6} spacing={3}>
          <Icon as={InfoIcon} color="#FF0032" boxSize="18px" />
          <Text color="#FFFFFF" fontWeight="700" fontSize="22px">API Reference</Text>
          <Text color="#444" fontSize="13px" pt="3px">TimeCast</Text>
        </HStack>

        {loading && <DocSkeleton />}

        {error && (
          <Box bg="#1A0A0A" border="1px solid #FF003244" borderRadius="10px" p={5}
               color="#FF6666" fontSize="14px">
            Ошибка загрузки документации: {error}
          </Box>
        )}

        {!loading && !error && (
          <Flex gap={8} align="flex-start">
            {/* Sticky TOC (≥ xl only) */}
            <TableOfContents headings={headings} activeId={activeId} />

            {/* Main content */}
            <Box flex={1} minW={0}>
              <Box
                bg="#141820"
                border="1px solid #2A2E36"
                borderRadius="14px"
                p={[5, 8]}
                sx={contentSx}
              >
                <MarkdownRenderer markdownText={content} />
              </Box>
            </Box>
          </Flex>
        )}

      </Box>
    </Flex>
  );
};

export default DocumentationPage;
