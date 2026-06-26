import React, { useState, useMemo, useEffect, useCallback, useRef } from "react";
import {
  HStack,
  Box,
  Textarea,
  Text,
  Flex,
  Badge,
  Wrap,
  WrapItem,
  Button,
  Icon,
  Tooltip,
  Divider,
  useToast,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  Kbd,
} from "@chakra-ui/react";
import {
  AttachmentIcon,
  ArrowUpIcon,
  DownloadIcon,
  DeleteIcon,
  TimeIcon,
  CheckIcon,
  RepeatClockIcon,
  CopyIcon,
  InfoOutlineIcon,
  LinkIcon,
} from "@chakra-ui/icons";
import { fetchZipUrl, uploadCSVFiles } from "../API/services/file_services";
import {
  sendClassicGraduate,
  sendNeiroGraduate,
  sendTimeSeriesGraduate,
  sendTimeSeriesInference,
  sendTimeSeriesNeiroGraduate,
  sendTimeSeriesNeiroInference,
} from "../API/services/graduate_services";
import { sendClassicInference, sendNeiroInference } from "../API/services/inference_services";
import { sendSeasonAnalytic } from "../API/services/season_analytic_services";
import {
  queueClassicGraduate,
  queueNeiroGraduate,
  queueTimeSeriesGraduate,
  queueTimeSeriesNeiroGraduate,
} from "../API/services/task_services";
import LogStreamComponent from "../API/apiLogStreamComponent";
import LogViewer from "../components/LogViewer";
import ForecastChart from "../components/ForecastChart";
import TrainingResults from "../components/TrainingResults";
import DecompositionChart from "../components/DecompositionChart";
import GenericSeriesForm from "../components/GenericSeriesForm";
import TaskStatusPanel from "../components/TaskStatusPanel";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const firstPeriodOf = (results) => {
  const firstItem = results && Object.values(results)[0];
  return firstItem && Object.values(firstItem)[0];
};
const isTrainingResults = (r) => "best_model" in (firstPeriodOf(r) || {});
const isDecompositionResults = (r) => "trend" in (firstPeriodOf(r) || {});

const detectOp = (jsonStr) => {
  try {
    const r = JSON.parse(jsonStr);
    if (r.proccess) return { label: "Декомпозиция", color: "purple" };
    const hasSrc = !!r.dataset?.source;
    if (hasSrc && r.graduate)
      return r.graduate.dictmodels
        ? { label: "Нейро обучение (generic)", color: "blue" }
        : { label: "Classic обучение (generic)", color: "cyan" };
    if (hasSrc && r.inference)
      return r.inference.dictmodels
        ? { label: "Нейро инференс (generic)", color: "blue" }
        : { label: "Classic инференс (generic)", color: "cyan" };
    if (r.inference?.dictmodels?.IFFT || r.inference?.dictmodels?.IF)
      return { label: "Нейро инференс (retail)", color: "orange" };
    if (r.inference) return { label: "Classic инференс (retail)", color: "yellow" };
    if (r.graduate && r.models_params) return { label: "Classic обучение (retail)", color: "yellow" };
    if (r.graduate) return { label: "Нейро обучение (retail)", color: "orange" };
    return { label: "Неизвестная операция", color: "red" };
  } catch {
    return { label: "Некорректный JSON", color: "red" };
  }
};

const RETAIL_TEMPLATES = {
  "Инференс (classic)": JSON.stringify(
    { dataset: { store_id: "STORE_1" },
      inference: { dictseasonal: { week: 7, month: 30 }, future_or_estimate: "estimate" } },
    null, 2
  ),
  "Инференс (нейро)": JSON.stringify(
    { dataset: { store_id: "STORE_1" },
      inference: { dictseasonal: { week: 7, month: 30 }, future_or_estimate: "estimate",
        dictmodels: { IFFT: { depth: 6, dim: 256, dim_head: 64, heads: 8,
          num_tokens_per_variate: 1, use_reversible_instance_norm: true } }, use_device: "cuda" } },
    null, 2
  ),
  "Обучение (classic)": JSON.stringify(
    { dataset: { store_id: "STORE_1" },
      graduate: { dictseasonal: { week: 7, month: 30 } },
      models_params: { AUTOARIMA: [3, 3, 0, 0, 1, 1, "week"] } },
    null, 2
  ),
  "Обучение (нейро)": JSON.stringify(
    { dataset: { store_id: "STORE_1" },
      graduate: { dictseasonal: { week: 7, month: 30 },
        dictmodels: { IFFT: { depth: 6, dim: 256, dim_head: 64, heads: 8,
          num_tokens_per_variate: 1, use_reversible_instance_norm: true } },
        use_device: "cuda" } },
    null, 2
  ),
};

// ---------------------------------------------------------------------------
// URL query param (share feature)
// ---------------------------------------------------------------------------

const readUrlQuery = () => {
  try {
    const hash = window.location.hash; // e.g. "#/query?q=..."
    const search = hash.includes("?") ? hash.split("?")[1] : "";
    const q = new URLSearchParams(search).get("q");
    if (!q) return null;
    const decoded = decodeURIComponent(q);
    JSON.parse(decoded); // validate
    return decoded;
  } catch {
    return null;
  }
};

// ---------------------------------------------------------------------------
// History helpers (localStorage)
// ---------------------------------------------------------------------------

const HISTORY_KEY = "timecast_query_history";
const HISTORY_MAX = 8;

const loadHistory = () => {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); }
  catch { return []; }
};

const saveToHistory = (query) => {
  try {
    const prev = loadHistory().filter((q) => q !== query);
    localStorage.setItem(HISTORY_KEY, JSON.stringify([query, ...prev].slice(0, HISTORY_MAX)));
  } catch {}
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

const SectionBox = ({ title, badge, badgeColor, headerRight, children, ...rest }) => (
  <Box
    border="1px solid #2A2E36"
    borderRadius="14px"
    overflow="hidden"
    bg="#141820"
    {...rest}
  >
    <HStack
      px={5}
      py={3}
      bg="#0F1218"
      borderBottom="1px solid #2A2E36"
      justify="space-between"
    >
      <HStack spacing={3}>
        <Text color="#FFFFFF" fontWeight="600" fontSize="15px">
          {title}
        </Text>
        {badge && (
          <Badge colorScheme={badgeColor ?? "gray"} fontSize="10px" px={2} borderRadius="6px">
            {badge}
          </Badge>
        )}
      </HStack>
      {headerRight}
    </HStack>
    <Box p={5}>{children}</Box>
  </Box>
);

const ToolbarButton = ({ icon, label, onClick, isLoading, colorScheme = "red", isDisabled }) => (
  <Tooltip label={label} placement="top" hasArrow>
    <Button
      size="sm"
      variant="outline"
      borderColor={colorScheme === "yellow" ? "#FFBF00" : colorScheme === "cyan" ? "#00B5D8" : "#FF0032"}
      color={colorScheme === "yellow" ? "#FFBF00" : colorScheme === "cyan" ? "#00B5D8" : "#FF0032"}
      _hover={{
        bg: colorScheme === "yellow" ? "#FFBF0018" : colorScheme === "cyan" ? "#00B5D818" : "#FF003218",
      }}
      leftIcon={<Icon as={icon} />}
      onClick={onClick}
      isLoading={isLoading}
      isDisabled={isDisabled}
      borderRadius="8px"
      h="36px"
      minW="120px"
    >
      {label}
    </Button>
  </Tooltip>
);

const IconBtn = ({ icon, label, onClick, color = "#555" }) => (
  <Tooltip label={label} placement="top" hasArrow>
    <Button
      size="xs"
      variant="ghost"
      color={color}
      _hover={{ color: "#FFFFFF", bg: "#2A2E36" }}
      onClick={onClick}
      p={1}
      minW="24px"
      h="24px"
      borderRadius="6px"
    >
      <Icon as={icon} boxSize="12px" />
    </Button>
  </Tooltip>
);

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const QueryPage = () => {
  const toast = useToast();
  const { isOpen: isHelpOpen, onOpen: onHelpOpen, onClose: onHelpClose } = useDisclosure();
  const resultsRef = useRef(null);
  const logRef = useRef(null);

  const [request, setRequest] = useState(() => {
    return readUrlQuery() || JSON.stringify(
      { dataset: { store_id: "STORE_1" },
        inference: { dictseasonal: { week: 7, month: 30, quater: 90 },
          dictmodels: { IFFT: { depth: 6, dim: 256, dim_head: 64, heads: 8,
            num_tokens_per_variate: 1, num_variates: 7, use_reversible_instance_norm: true } },
          future_or_estimate: "estimate", use_device: "cuda" } },
      null, 2
    );
  });
  const [responseText, setResponseText] = useState("");
  const [files, setFiles] = useState([]);
  const [resultData, setResultData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [queueTaskId, setQueueTaskId] = useState(null);
  const [history, setHistory] = useState(loadHistory);
  const [showHistory, setShowHistory] = useState(false);

  const detectedOp = useMemo(() => detectOp(request), [request]);

  const isJsonValid = useMemo(() => {
    try { JSON.parse(request); return true; }
    catch { return false; }
  }, [request]);

  // Auto-scroll log textarea to bottom when new content arrives
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [responseText]);

  const pushHistory = useCallback((q) => {
    saveToHistory(q);
    setHistory(loadHistory());
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.key === "Enter" && e.ctrlKey && !e.shiftKey && !isLoading) {
        e.preventDefault();
        handleSendQuery();
      }
      if (e.key === "Enter" && e.ctrlKey && e.shiftKey) {
        e.preventDefault();
        handleQueueTraining();
      }
      if (e.key === "Escape") setShowHistory(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoading, request]);

  // --- Utility handlers ---

  const handlePrettify = () => {
    try {
      setRequest(JSON.stringify(JSON.parse(request), null, 2));
    } catch {
      toast({ title: "Некорректный JSON", status: "error", duration: 2000, position: "bottom-right" });
    }
  };

  const handleCopyJson = () => {
    navigator.clipboard.writeText(request).then(() => {
      toast({ title: "JSON скопирован", status: "success", duration: 1500, isClosable: true, position: "bottom-right" });
    });
  };

  const handleCopyResponse = () => {
    if (!responseText) return;
    navigator.clipboard.writeText(responseText).then(() => {
      toast({ title: "Ответ скопирован", status: "success", duration: 1500, isClosable: true, position: "bottom-right" });
    });
  };

  const handleDownloadResponse = () => {
    if (!responseText) return;
    const trimmed = responseText.trim();
    const isJson = trimmed.startsWith("{") || trimmed.startsWith("[");
    const ext  = isJson ? "json" : "txt";
    const type = isJson ? "application/json" : "text/plain";
    const blob = new Blob([responseText], { type });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url;
    a.download = `timecast-response.${ext}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleShare = () => {
    try {
      const hash = window.location.hash.split("?")[0] || "#/query";
      const url = `${window.location.origin}${window.location.pathname}${hash}?q=${encodeURIComponent(request)}`;
      navigator.clipboard.writeText(url);
      toast({ title: "Ссылка скопирована", status: "success", duration: 2500, position: "bottom-right" });
    } catch {
      toast({ title: "Ошибка копирования ссылки", status: "error", duration: 2000, position: "bottom-right" });
    }
  };

  // --- Main handlers ---

  const handleQueueTraining = async () => {
    try {
      const p = JSON.parse(request);
      let resp;
      const hasSrc = !!p.dataset?.source;
      if (hasSrc && p.graduate?.dictmodels) resp = await queueTimeSeriesNeiroGraduate(p);
      else if (hasSrc && p.graduate) resp = await queueTimeSeriesGraduate(p);
      else if (p.graduate && p.models_params) resp = await queueClassicGraduate(p);
      else if (p.graduate) resp = await queueNeiroGraduate(p);
      else {
        toast({ title: "Только операции обучения можно ставить в очередь", status: "warning", duration: 3000, position: "bottom-right" });
        return;
      }
      setQueueTaskId(resp.task_id);
      toast({
        title: "Задача добавлена в очередь",
        description: resp.task_id,
        status: "success",
        duration: 4000,
        isClosable: true,
        position: "bottom-right",
      });
    } catch (err) {
      toast({ title: "Ошибка постановки в очередь", description: err.message, status: "error", duration: 4000, isClosable: true, position: "bottom-right" });
    }
  };

  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files);
    if (selected.length === 3) {
      setFiles(selected);
    } else {
      toast({ title: "Выберите ровно 3 CSV-файла", status: "warning", duration: 3000, position: "bottom-right" });
    }
  };

  const handleSendCSV = async () => {
    if (files.length !== 3) {
      toast({ title: "Выберите ровно 3 CSV-файла", status: "warning", duration: 3000, position: "bottom-right" });
      return;
    }
    try {
      await uploadCSVFiles(files);
      toast({ title: "CSV файлы загружены", status: "success", duration: 3000, isClosable: true, position: "bottom-right" });
    } catch {
      toast({ title: "Ошибка загрузки CSV", status: "error", duration: 3000, isClosable: true, position: "bottom-right" });
    }
  };

  const handleSendQuery = async () => {
    setResultData(null);
    setIsLoading(true);
    try {
      const p = JSON.parse(request);
      let response;

      if (p.dataset?.source && p.inference) {
        response = await (p.inference.dictmodels
          ? sendTimeSeriesNeiroInference(p)
          : sendTimeSeriesInference(p));
      } else if (p.inference) {
        response = await (p.inference.dictmodels?.IFFT || p.inference.dictmodels?.IF
          ? sendNeiroInference(p)
          : sendClassicInference(p));
      } else if (p.dataset?.source && p.graduate) {
        response = await (p.graduate.dictmodels
          ? sendTimeSeriesNeiroGraduate(p)
          : sendTimeSeriesGraduate(p));
      } else if (p.dataset && p.graduate) {
        response = await (p.models_params ? sendClassicGraduate(p) : sendNeiroGraduate(p));
      } else if (p.proccess) {
        response = await sendSeasonAnalytic(p);
      } else {
        toast({ title: "Некорректная структура запроса", status: "warning", duration: 3000, position: "bottom-right" });
        return;
      }
      setResponseText(JSON.stringify(response, null, 2));
      setResultData(response);
      pushHistory(request);
      toast({ title: "Запрос выполнен", status: "success", duration: 2000, position: "bottom-right" });
      // Scroll to results after render
      requestAnimationFrame(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    } catch (error) {
      const msg = error.message || "Запрос не выполнен.";
      setResponseText(`Ошибка: ${msg}`);
      toast({ title: "Ошибка запроса", description: msg, status: "error", duration: 5000, isClosable: true, position: "bottom-right" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleTextareaKeyDown = (e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const ta = e.target;
      const { selectionStart: s, selectionEnd: end } = ta;
      const newVal = request.substring(0, s) + "  " + request.substring(end);
      setRequest(newVal);
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = s + 2;
      });
    }
  };

  const handleDownloadArchive = async () => {
    try {
      const url = await fetchZipUrl();
      window.location.href = url;
      toast({ title: "Архив загружен", status: "success", duration: 2000, position: "bottom-right" });
    } catch {
      toast({ title: "Ошибка загрузки архива", status: "error", duration: 3000, position: "bottom-right" });
    }
  };

  const handleNewLog = (newLog) =>
    setResponseText((prev) => prev + "\n" + newLog);

  // ---------------------------------------------------------------------------
  return (
    <Flex
      direction="column"
      w="100%"
      px={[4, 6, 8]}
      pt={6}
      pb={10}
      gap={5}
      flexGrow={1}
      align="center"
    >
      <Box w="100%" maxW="1400px">

        {/* Top two columns — stacks on small screens */}
        <Flex gap={5} align="stretch" mb={5} direction={["column", "column", "row"]}>

          {/* Left — Query builder */}
          <SectionBox
            title="Запрос"
            badge={detectedOp.label}
            badgeColor={detectedOp.color}
            flex={1}
          >
            <GenericSeriesForm onBuild={setRequest} />

            <Divider borderColor="#2A2E36" my={4} />

            <Text fontSize="12px" color="#555" mb={2} fontWeight="600" letterSpacing="0.08em">
              RETAIL ШАБЛОНЫ
            </Text>
            <Wrap spacing={2} mb={4}>
              {Object.entries(RETAIL_TEMPLATES).map(([label, tpl]) => (
                <WrapItem key={label}>
                  <Button
                    size="xs"
                    variant="outline"
                    borderColor="#FF003255"
                    color="#FF0032AA"
                    _hover={{ borderColor: "#FF0032", color: "#FF0032", bg: "#FF003210" }}
                    onClick={() => setRequest(tpl)}
                    borderRadius="6px"
                  >
                    {label}
                  </Button>
                </WrapItem>
              ))}
            </Wrap>

            {/* JSON label row */}
            <HStack justify="space-between" mb={2}>
              <HStack spacing={2} align="center">
                <Tooltip label={isJsonValid ? "JSON валиден" : "Ошибка JSON"} placement="top" hasArrow>
                  <Box
                    w="7px"
                    h="7px"
                    borderRadius="full"
                    bg={isJsonValid ? "#48BB78" : "#FF8888"}
                    flexShrink={0}
                    cursor="default"
                  />
                </Tooltip>
                <Text fontSize="12px" color="#555" fontWeight="600" letterSpacing="0.08em">
                  JSON
                </Text>
              </HStack>

              <HStack spacing={1}>
                <Text fontSize="11px" color="#333">Ctrl+Enter</Text>
                <IconBtn icon={CopyIcon} label="Скопировать JSON" onClick={handleCopyJson} />
                <Tooltip label="Форматировать JSON" placement="top" hasArrow>
                  <Button
                    size="xs"
                    variant="ghost"
                    color="#555"
                    _hover={{ color: "#FFBF00", bg: "#2A2E36" }}
                    onClick={handlePrettify}
                    h="24px"
                    px={2}
                    borderRadius="6px"
                    fontSize="11px"
                    fontFamily="monospace"
                  >
                    {"{ }"}
                  </Button>
                </Tooltip>
                {history.length > 0 && (
                  <Tooltip label="История запросов" placement="top" hasArrow>
                    <Button
                      size="xs"
                      variant="ghost"
                      color={showHistory ? "#FFBF00" : "#555"}
                      _hover={{ color: "#FFBF00" }}
                      leftIcon={<Icon as={RepeatClockIcon} boxSize="11px" />}
                      onClick={() => setShowHistory((v) => !v)}
                      h="24px"
                      px={2}
                      borderRadius="6px"
                      fontSize="11px"
                    >
                      История
                    </Button>
                  </Tooltip>
                )}
              </HStack>
            </HStack>

            {/* History dropdown */}
            {showHistory && history.length > 0 && (
              <Box
                bg="#0D1017"
                border="1px solid #2A2E36"
                borderRadius="10px"
                mb={2}
                maxH="180px"
                overflowY="auto"
              >
                {history.map((q, i) => (
                  <Box
                    key={i}
                    px={3}
                    py={2}
                    cursor="pointer"
                    borderBottom={i < history.length - 1 ? "1px solid #1A1D21" : "none"}
                    _hover={{ bg: "#141820" }}
                    onClick={() => { setRequest(q); setShowHistory(false); }}
                  >
                    <Text color="#888" fontSize="11px" fontFamily="monospace" noOfLines={1}>
                      {q.replace(/\s+/g, " ").slice(0, 80)}
                    </Text>
                  </Box>
                ))}
              </Box>
            )}

            <Textarea
              value={request}
              onChange={(e) => setRequest(e.target.value)}
              onKeyDown={handleTextareaKeyDown}
              h="220px"
              bg="#0D1017"
              color="#E8E8E8"
              border="1px solid"
              borderColor={isJsonValid ? "#2A2E36" : "#FF003255"}
              _hover={{ borderColor: isJsonValid ? "#444" : "#FF0032AA" }}
              _focus={{ borderColor: "#FF0032", boxShadow: "0 0 0 1px #FF003244" }}
              borderRadius="10px"
              resize="none"
              fontFamily="monospace"
              fontSize="13px"
            />
          </SectionBox>

          {/* Right — Live logs */}
          <SectionBox
            title="Лог выполнения"
            flex={1}
            headerRight={
              responseText ? (
                <HStack spacing={1}>
                  <IconBtn icon={DownloadIcon} label="Скачать ответ" onClick={handleDownloadResponse} color="#444" />
                  <IconBtn icon={CopyIcon}     label="Скопировать ответ" onClick={handleCopyResponse} color="#444" />
                </HStack>
              ) : undefined
            }
          >
            <LogStreamComponent onNewLog={handleNewLog} />
            <LogViewer ref={logRef} value={responseText} height="480px" />
          </SectionBox>
        </Flex>

        {/* Toolbar */}
        <Box
          bg="#0F1218"
          border="1px solid #2A2E36"
          borderRadius="12px"
          px={5}
          py={4}
          mb={5}
          overflowX="auto"
          sx={{
            "&::-webkit-scrollbar": { h: "4px" },
            "&::-webkit-scrollbar-track": { bg: "transparent" },
            "&::-webkit-scrollbar-thumb": { bg: "#2A2E36", borderRadius: "2px" },
          }}
        >
          <HStack spacing={3} minW="max-content">
            {/* File upload */}
            <Box>
              <input
                id="file-upload"
                type="file"
                accept=".csv"
                multiple
                onChange={handleFileChange}
                style={{ display: "none" }}
              />
              <label htmlFor="file-upload">
                <Button
                  as="span"
                  size="sm"
                  variant="outline"
                  borderColor="#2A2E36"
                  color="#888"
                  _hover={{ borderColor: "#888", color: "#FFFFFF" }}
                  leftIcon={<Icon as={AttachmentIcon} />}
                  borderRadius="8px"
                  h="36px"
                  cursor="pointer"
                >
                  {files.length === 3 ? (
                    <HStack spacing={1}>
                      <Icon as={CheckIcon} color="#48BB78" boxSize="12px" />
                      <Text fontSize="12px">3 файла выбраны</Text>
                    </HStack>
                  ) : (
                    "Выбрать CSV (3)"
                  )}
                </Button>
              </label>
            </Box>

            <Divider orientation="vertical" h="30px" borderColor="#2A2E36" />

            <ToolbarButton
              icon={ArrowUpIcon}
              label="Send CSV"
              onClick={handleSendCSV}
              colorScheme="red"
            />
            <ToolbarButton
              icon={ArrowUpIcon}
              label="Send Query"
              onClick={handleSendQuery}
              isLoading={isLoading}
              isDisabled={isLoading}
              colorScheme="red"
            />
            <ToolbarButton
              icon={DownloadIcon}
              label="Load Zip"
              onClick={handleDownloadArchive}
              colorScheme="cyan"
            />

            <Divider orientation="vertical" h="30px" borderColor="#2A2E36" />

            <ToolbarButton
              icon={TimeIcon}
              label="В очередь"
              onClick={handleQueueTraining}
              colorScheme="yellow"
            />

            <Box flex={1} />

            <Tooltip label="Поделиться запросом" placement="top" hasArrow>
              <Button
                size="sm"
                variant="ghost"
                color="#555"
                _hover={{ color: "#FFBF00", bg: "#FFBF0018" }}
                leftIcon={<Icon as={LinkIcon} boxSize="12px" />}
                onClick={handleShare}
                h="36px"
                px={3}
                borderRadius="8px"
                fontSize="13px"
              >
                Поделиться
              </Button>
            </Tooltip>

            {responseText && (
              <Tooltip label="Очистить лог" placement="top" hasArrow>
                <Button
                  size="sm"
                  variant="ghost"
                  color="#555"
                  _hover={{ color: "#FF0032" }}
                  leftIcon={<Icon as={DeleteIcon} />}
                  onClick={() => setResponseText("")}
                  h="36px"
                >
                  Очистить
                </Button>
              </Tooltip>
            )}

            <Tooltip label="Клавиатурные шорткаты" placement="top" hasArrow>
              <Button
                size="sm"
                variant="ghost"
                color="#444"
                _hover={{ color: "#FFFFFF" }}
                onClick={onHelpOpen}
                h="36px"
                px={2}
              >
                <Icon as={InfoOutlineIcon} boxSize="14px" />
              </Button>
            </Tooltip>
          </HStack>
        </Box>

        {/* Results */}
        <Box ref={resultsRef}>
          {resultData?.results && (
            isDecompositionResults(resultData.results) ? (
              <DecompositionChart results={resultData.results} />
            ) : isTrainingResults(resultData.results) ? (
              <TrainingResults results={resultData.results} />
            ) : (
              <ForecastChart results={resultData.results} />
            )
          )}

          {queueTaskId && (
            <TaskStatusPanel
              taskId={queueTaskId}
              onResultReady={(result) => setResultData(result)}
            />
          )}
        </Box>

      </Box>

      {/* Keyboard shortcuts modal */}
      <Modal isOpen={isHelpOpen} onClose={onHelpClose} isCentered size="md">
        <ModalOverlay bg="blackAlpha.700" backdropFilter="blur(6px)" />
        <ModalContent bg="#141820" border="1px solid #2A2E36" borderRadius="14px">
          <ModalHeader
            color="#FFFFFF"
            fontSize="15px"
            fontWeight="600"
            borderBottom="1px solid #2A2E36"
            pb={3}
          >
            Клавиатурные шорткаты
          </ModalHeader>
          <ModalCloseButton color="#555" top={3} right={4} />
          <ModalBody py={5}>
            <Flex direction="column" gap={3}>
              {[
                { keys: ["Ctrl", "Enter"],         desc: "Отправить запрос" },
                { keys: ["Ctrl", "Shift", "Enter"], desc: "Поставить в очередь" },
                { keys: ["Tab"],                    desc: "Вставить 2 пробела (в JSON)" },
                { keys: ["Esc"],                    desc: "Закрыть историю запросов" },
              ].map(({ keys, desc }) => (
                <HStack key={desc} justify="space-between">
                  <HStack spacing={1}>
                    {keys.map((k, i) => (
                      <React.Fragment key={k}>
                        <Kbd
                          bg="#0D1017"
                          color="#AAAAAA"
                          border="1px solid #2A2E36"
                          borderRadius="6px"
                          fontSize="11px"
                          px={2}
                          py="2px"
                        >
                          {k}
                        </Kbd>
                        {i < keys.length - 1 && (
                          <Text color="#444" fontSize="11px">+</Text>
                        )}
                      </React.Fragment>
                    ))}
                  </HStack>
                  <Text color="#888" fontSize="13px">{desc}</Text>
                </HStack>
              ))}
            </Flex>

            <Divider borderColor="#2A2E36" my={4} />

            <Text color="#444" fontSize="11px">
              JSON-лог автоматически определяет формат: цветной лог или подсветка синтаксиса JSON
            </Text>
          </ModalBody>
        </ModalContent>
      </Modal>

    </Flex>
  );
};

export default QueryPage;
