import React, { useState, useMemo } from "react";
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
} from "@chakra-ui/react";
import {
  AttachmentIcon,
  ArrowUpIcon,
  DownloadIcon,
  DeleteIcon,
  TimeIcon,
  CheckIcon,
} from "@chakra-ui/icons";
import useWindowDimensions from "../hooks/window_dimensions";
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
// Sub-components
// ---------------------------------------------------------------------------

const SectionBox = ({ title, badge, badgeColor, children, ...rest }) => (
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
      spacing={3}
    >
      <Text color="#FFFFFF" fontWeight="600" fontSize="15px">
        {title}
      </Text>
      {badge && (
        <Badge colorScheme={badgeColor ?? "gray"} fontSize="10px" px={2} borderRadius="6px">
          {badge}
        </Badge>
      )}
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

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const QueryPage = () => {
  const { width } = useWindowDimensions();

  const [request, setRequest] = useState(
    JSON.stringify(
      { dataset: { store_id: "STORE_1" },
        inference: { dictseasonal: { week: 7, month: 30, quater: 90 },
          dictmodels: { IFFT: { depth: 6, dim: 256, dim_head: 64, heads: 8,
            num_tokens_per_variate: 1, num_variates: 7, use_reversible_instance_norm: true } },
          future_or_estimate: "estimate", use_device: "cuda" } },
      null, 2
    )
  );
  const [responseText, setResponseText] = useState("");
  const [files, setFiles] = useState([]);
  const [resultData, setResultData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [queueTaskId, setQueueTaskId] = useState(null);

  const detectedOp = useMemo(() => detectOp(request), [request]);

  // --- Handlers ---

  const handleQueueTraining = async () => {
    try {
      const p = JSON.parse(request);
      let resp;
      const hasSrc = !!p.dataset?.source;
      if (hasSrc && p.graduate?.dictmodels) resp = await queueTimeSeriesNeiroGraduate(p);
      else if (hasSrc && p.graduate) resp = await queueTimeSeriesGraduate(p);
      else if (p.graduate && p.models_params) resp = await queueClassicGraduate(p);
      else if (p.graduate) resp = await queueNeiroGraduate(p);
      else { setResponseText("Очередь доступна только для операций обучения (graduate)."); return; }
      setQueueTaskId(resp.task_id);
      setResponseText(`Задача поставлена в очередь: ${resp.task_id}`);
    } catch (err) {
      setResponseText(`Ошибка: ${err.message}`);
    }
  };

  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files);
    if (selected.length === 3) setFiles(selected);
    else setResponseText("Выберите ровно 3 CSV-файла.");
  };

  const handleSendCSV = async () => {
    if (files.length !== 3) { setResponseText("Выберите ровно 3 CSV-файла."); return; }
    try {
      const msg = await uploadCSVFiles(files);
      setResponseText(msg);
    } catch {
      setResponseText("Ошибка загрузки CSV.");
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
        setResponseText("Некорректная структура запроса.");
        return;
      }
      setResponseText(JSON.stringify(response, null, 2));
      setResultData(response);
    } catch (error) {
      setResponseText(`Ошибка: ${error.message || "Запрос не выполнен."}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadArchive = async () => {
    try {
      const url = await fetchZipUrl();
      window.location.href = url;
      setResponseText("Архив загружен.");
    } catch {
      setResponseText("Ошибка загрузки архива.");
    }
  };

  const handleNewLog = (newLog) =>
    setResponseText((prev) => prev + "\n" + newLog);

  // ---------------------------------------------------------------------------
  return (
    <Flex
      direction="column"
      w={width}
      px={[4, 6, 8]}
      pt={6}
      pb={10}
      gap={5}
      flexGrow={1}
      align="center"
    >
      <Box w="100%" maxW="1400px">

        {/* Top two columns */}
        <HStack spacing={5} align="stretch" mb={5}>

          {/* Left — Query builder */}
          <SectionBox title="Запрос" badge={detectedOp.label} badgeColor={detectedOp.color} w="50%">
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

            <Text fontSize="12px" color="#555" mb={2} fontWeight="600" letterSpacing="0.08em">
              JSON
            </Text>
            <Textarea
              value={request}
              onChange={(e) => setRequest(e.target.value)}
              h="220px"
              bg="#0D1017"
              color="#E8E8E8"
              border="1px solid #2A2E36"
              _hover={{ borderColor: "#444" }}
              _focus={{ borderColor: "#FF0032", boxShadow: "0 0 0 1px #FF003244" }}
              borderRadius="10px"
              resize="none"
              fontFamily="monospace"
              fontSize="13px"
            />
          </SectionBox>

          {/* Right — Live logs */}
          <SectionBox title="Лог выполнения" w="50%">
            <LogStreamComponent onNewLog={handleNewLog} />
            <Textarea
              value={responseText}
              onChange={(e) => setResponseText(e.target.value)}
              h="480px"
              bg="#0D1017"
              color="#AAAAAA"
              border="1px solid #2A2E36"
              _hover={{ borderColor: "#444" }}
              _focus={{ borderColor: "#444", boxShadow: "none" }}
              borderRadius="10px"
              resize="none"
              fontFamily="monospace"
              fontSize="12px"
            />
          </SectionBox>
        </HStack>

        {/* Toolbar */}
        <Box
          bg="#0F1218"
          border="1px solid #2A2E36"
          borderRadius="12px"
          px={5}
          py={4}
          mb={5}
        >
          <HStack spacing={3} flexWrap="wrap" gap={3}>
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
                  ml="auto"
                >
                  Очистить
                </Button>
              </Tooltip>
            )}
          </HStack>
        </Box>

        {/* Results */}
        {resultData?.results && (
          isDecompositionResults(resultData.results) ? (
            <DecompositionChart results={resultData.results} />
          ) : isTrainingResults(resultData.results) ? (
            <TrainingResults results={resultData.results} />
          ) : (
            <ForecastChart results={resultData.results} />
          )
        )}

        {/* Background task panel */}
        {queueTaskId && (
          <TaskStatusPanel
            taskId={queueTaskId}
            onResultReady={(result) => setResultData(result)}
          />
        )}
      </Box>
    </Flex>
  );
};

export default QueryPage;
