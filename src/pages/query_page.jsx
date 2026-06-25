import React, { useState, useMemo } from "react";
import { VStack, HStack, Box, Textarea, Text, Flex, Badge, Wrap, WrapItem, Button } from "@chakra-ui/react";
import MenuActiveComponent from "../components/MenuActiveComponent";
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

// Различаем форму ответа по первому периоду первого item.
const firstPeriodOf = (results) => {
  const firstItem = results && Object.values(results)[0];
  return firstItem && Object.values(firstItem)[0];
};
const isTrainingResults = (results) => "best_model" in (firstPeriodOf(results) || {});
const isDecompositionResults = (results) => "trend" in (firstPeriodOf(results) || {});

// Определяет читаемое название операции по структуре JSON-запроса.
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

// Шаблоны запросов для retail-операций.
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

const QueryPage = () => {
  const { width } = useWindowDimensions();

  // State hooks inside the component
  const [request, setRequest] = useState(`{
        "dataset": {
            "store_id": "STORE_1"
        },
        "inference": {
            "dictseasonal": {
                "week": 7,
                "month": 30,
                "quater": 90
            },
            "dictmodels": {
                "IFFT": {
                    "depth": 6,
                    "dim": 256,
                    "dim_head": 64,
                    "heads": 8,
                    "num_tokens_per_variate": 1,
                    "num_variates": 7,
                    "use_reversible_instance_norm": true
                }
            },
            "future_or_estimate": "estimate",
            "use_device": "cuda"
        }
    }`);
  const [responseText, setResponseText] = useState("");
  const [files, setFiles] = useState([]);
  const [resultData, setResultData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [queueTaskId, setQueueTaskId] = useState(null);

  const detectedOp = useMemo(() => detectOp(request), [request]);

  // Отправляет обучение в фоновую очередь и немедленно возвращает task_id.
  const handleQueueTraining = async () => {
    try {
      const parsedRequest = JSON.parse(request);
      let resp;
      const hasSrc = !!parsedRequest.dataset?.source;
      if (hasSrc && parsedRequest.graduate?.dictmodels) {
        resp = await queueTimeSeriesNeiroGraduate(parsedRequest);
      } else if (hasSrc && parsedRequest.graduate) {
        resp = await queueTimeSeriesGraduate(parsedRequest);
      } else if (parsedRequest.graduate && parsedRequest.models_params) {
        resp = await queueClassicGraduate(parsedRequest);
      } else if (parsedRequest.graduate) {
        resp = await queueNeiroGraduate(parsedRequest);
      } else {
        setResponseText("Очередь доступна только для операций обучения (graduate).");
        return;
      }
      setQueueTaskId(resp.task_id);
      setResponseText(`Задача поставлена в очередь: ${resp.task_id}`);
    } catch (err) {
      setResponseText(`Ошибка: ${err.message}`);
    }
  };

  // Function to handle CSV file selection
  const handleFileChange = (e) => {
    const selectedFiles = e.target.files;
    if (selectedFiles.length === 3) {
      setFiles(Array.from(selectedFiles)); // Update state with selected files
    } else {
      setResponseText("Please select exactly 3 files.");
    }
  };

  // Function for sending CSV files to the server
  const handleSendCSV = async () => {
    console.log("Send CSV button clicked");
    if (files.length !== 3) {
      setResponseText("Please select exactly 3 CSV files.");
      return;
    }

    try {
      console.log("Sending a CSV");
      const responseMessage = await uploadCSVFiles(files); // Send files to the server
      setResponseText(responseMessage); // Display server response
    } catch (error) {
      setResponseText("Error while uploading CSV files.");
    }
  };

  // Function to handle sending the query
  const handleSendQuery = async () => {
    setResultData(null);
    setIsLoading(true);
    try {
      const parsedRequest = JSON.parse(request);

      if (parsedRequest.dataset && parsedRequest.dataset.source && parsedRequest.inference) {
        // Обобщённый ряд (tidy CSV): нейросеть, если dictmodels, иначе classic.
        const isNeiro = !!parsedRequest.inference.dictmodels;
        console.log(`Sending TimeSeries (generic) ${isNeiro ? "Neiro" : "Classic"} Inference request...`);
        const response = isNeiro
          ? await sendTimeSeriesNeiroInference(parsedRequest)
          : await sendTimeSeriesInference(parsedRequest);
        setResponseText(JSON.stringify(response, null, 2));
        setResultData(response);
      } else if (parsedRequest.inference) {
        if (
          parsedRequest.inference.dictmodels &&
          (parsedRequest.inference.dictmodels.IFFT || parsedRequest.inference.dictmodels.IF)
        ) {
          console.log("Sending Neiro Inference request...");
          const response = await sendNeiroInference(parsedRequest);
          setResponseText(JSON.stringify(response, null, 2));
          setResultData(response);
        } else {
          console.log("Sending Classic Inference request...");
          const response = await sendClassicInference(parsedRequest);
          setResponseText(JSON.stringify(response, null, 2));
          setResultData(response);
        }
      } else if (parsedRequest.dataset && parsedRequest.dataset.source && parsedRequest.graduate) {
        // Обобщённый ряд (tidy CSV): нейросеть, если dictmodels, иначе classic (models_params).
        const isNeiro = !!parsedRequest.graduate.dictmodels;
        console.log(`Sending TimeSeries (generic) ${isNeiro ? "Neiro" : "Classic"} Graduate request...`);
        const response = isNeiro
          ? await sendTimeSeriesNeiroGraduate(parsedRequest)
          : await sendTimeSeriesGraduate(parsedRequest);
        setResponseText(JSON.stringify(response, null, 2));
        setResultData(response);
      } else if (parsedRequest.dataset && parsedRequest.graduate) {
        if (parsedRequest.models_params) {
          console.log("Sending Classic Graduate request...");
          const response = await sendClassicGraduate(parsedRequest);
          setResponseText(JSON.stringify(response, null, 2));
          setResultData(response);
        } else {
          console.log("Sending Neiro Graduate request...");
          const response = await sendNeiroGraduate(parsedRequest);
          setResponseText(JSON.stringify(response, null, 2));
          setResultData(response);
        }
      } else if (parsedRequest.proccess) {
        console.log("Sending Season Analytic request...");
        const response = await sendSeasonAnalytic(parsedRequest);
        setResponseText(JSON.stringify(response, null, 2));
        setResultData(response);
      } else {
        setResponseText("Invalid Query structure. Please check the format.");
      }
    } catch (error) {
      console.error("Failed to send query:", error);
      setResponseText(`Ошибка: ${error.message || "Запрос не выполнен."}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to handle downloading a ZIP file
  const handleDownloadArchive = async () => {
    try {
      console.log("Downloading archive...");
      const url = await fetchZipUrl(files); // Assuming this function returns the download URL for the ZIP file
      // Trigger the download by setting the location.href to the URL
      window.location.href = url;
      setResponseText("Zip was successfuly loaded"); // Display server response
    } catch (error) {
      console.error("Download error:", error); // Log detailed error information
      setResponseText("Error while downloading the ZIP file."); // Update the error message for download
    }
  };

  // �������, ������� ��������� �������� ��������� ������� � ������ ������
  const handleNewLog = (newLog) => {
    setResponseText((prevRequest) => prevRequest + "\n" + newLog); // ��������� ����� ��� � ������������� ������
  };

  return (
    <Flex
      direction="column"
      bg="transparent"
      padding={5}
      spacing="5px"
      flexGrow={1}
      align="center"
      justify="flex-start"
      width={width}
      height="100%"
      overflowX="hidden"
      overflowY="auto"
      paddingTop="120px"
    >
      <VStack
        minH="80%"
        minW="80%"
        align="stretch"
        bg="transparent"
        padding={5}
        spacing="5px"
        flexGrow={1}
      >
        <HStack spacing={10} align="stretch">
          {/* Left Column with Request */}
          <Box border="2px solid #FF0032" borderRadius="10px" p={5} w="50%" minH="600px">
            <Text fontSize="24px" fontWeight="bold" color="#FFFFFF" mb={4}>
              Insert Query
            </Text>
            <GenericSeriesForm onBuild={setRequest} />

            {/* Retail-шаблоны */}
            <Text fontSize="13px" color="#AAA" mt={2} mb={1}>
              Retail шаблоны:
            </Text>
            <Wrap spacing={2} mb={3}>
              {Object.entries(RETAIL_TEMPLATES).map(([label, tpl]) => (
                <WrapItem key={label}>
                  <Button
                    size="xs"
                    variant="outline"
                    borderColor="#FF0032"
                    color="#FF0032"
                    _hover={{ bg: "#FF003222" }}
                    onClick={() => setRequest(tpl)}
                  >
                    {label}
                  </Button>
                </WrapItem>
              ))}
            </Wrap>

            {/* Бейдж текущей операции */}
            <HStack mb={2}>
              <Text fontSize="12px" color="#888">
                Операция:
              </Text>
              <Badge colorScheme={detectedOp.color} fontSize="12px" px={2} borderRadius="6px">
                {detectedOp.label}
              </Badge>
            </HStack>

            <Textarea
              value={request}
              onChange={(e) => setRequest(e.target.value)}
              height="260px"
              bg="#2D2D2D"
              color="#FFFFFF"
              borderColor="#FF0032"
              resize="none"
            />
          </Box>

          {/* Right Column with Logs Stream */}
          <Box
            border="2px solid #FF0032"
            borderRadius="10px"
            p={5}
            w="50%"
            minH="600px"
            display="flex"
            flexDirection="column" // Ensure proper flex alignment
            alignItems="stretch" // Ensure elements stretch vertically
            justifyContent="flex-start" // Align content to top
            bg="transparent"
          >
            <Text fontSize="24px" fontWeight="bold" color="#FFFFFF" mb={4}>
              LogStream
            </Text>

            {/* ��������� LogStreamComponent ��� ��������� ����� ����� */}
            <LogStreamComponent onNewLog={handleNewLog} />

            {/* Ensure Textarea uses all available height */}
            <Textarea
              value={responseText} // ������ �� ���������� ��������� request
              onChange={(e) => setResponseText(e.target.value)} // ��������� ��������� ��� ���������
              height="500px"
              bg="#2D2D2D"
              color="#FFFFFF"
              borderColor="#FF0032"
              resize="none"
            />
          </Box>
        </HStack>

        <HStack spacing={0} align="center">
          <Box>
            {/* Custom file input button */}
            <label
              htmlFor="file-upload"
              style={{
                fontSize: "Inter",
                fontWeight: "0",
                width: "200px",
                height: "44px",
                backgroundColor: "#FF0032",
                color: "#FFFFFF",
                borderRadius: "10px",
                display: "inline-block",
                textAlign: "center",
                lineHeight: "44px",
                cursor: "pointer",
                transition: "background-color 0.2s, transform 0.1s",
              }}
            >
              Choose Files
            </label>

            {/* Hidden file input */}
            <input
              id="file-upload"
              type="file"
              accept=".csv"
              multiple
              onChange={handleFileChange}
              style={{ display: "none" }} // Hide the default input
            />

            <Text color="white" mt={2}>
              {files.length === 3 ? `Selected ${files.length} files` : "Please select 3 CSV files"}
            </Text>
          </Box>

          {/* Menu Component with Buttons */}
          <MenuActiveComponent
            isHorizontal={true}
            buttons={[
              { label: "Send CSV", path: "#" },
              { label: "Send Query", path: "#" },
              { label: "Load Zip", path: "#" },
            ]}
            showTitle={false}
            hideButtons={false}
            buttonWidth="200px"
            buttonHeight="44px"
            disabledLabels={isLoading ? ["Send Query"] : []}
            onClickActions={{
              "Send CSV": handleSendCSV,
              "Send Query": handleSendQuery,
              "Load Zip": handleDownloadArchive,
            }}
          />
          {/* Кнопка фоновой очереди — только для graduate-операций */}
          <Button
            size="sm"
            variant="outline"
            borderColor="#FFBF00"
            color="#FFBF00"
            _hover={{ bg: "#FFBF0022" }}
            onClick={handleQueueTraining}
            ml={2}
          >
            В очередь
          </Button>
          {isLoading && (
            <Text color="#AAA" fontSize="14px" mt={2}>
              Выполняется запрос…
            </Text>
          )}
        </HStack>

        {/* Дашборд: декомпозиция -> trend/seasonal/resid; обучение -> таблица; инференс -> прогноз */}
        {resultData &&
          resultData.results &&
          (isDecompositionResults(resultData.results) ? (
            <DecompositionChart results={resultData.results} />
          ) : isTrainingResults(resultData.results) ? (
            <TrainingResults results={resultData.results} />
          ) : (
            <ForecastChart results={resultData.results} />
          ))}

        {/* Панель фоновой задачи обучения */}
        {queueTaskId && (
          <TaskStatusPanel
            taskId={queueTaskId}
            onResultReady={(result) => setResultData(result)}
          />
        )}
      </VStack>
    </Flex>
  );
};

export default QueryPage;
