import React, { useState } from "react";
import { VStack, HStack, Box, Textarea, Text, Flex } from "@chakra-ui/react";
import MenuActiveComponent from "../components/MenuActiveComponent";
import useWindowDimensions from "../hooks/window_dimensions";
import { fetchZipUrl, uploadCSVFiles } from "../API/services/file_services";
import { sendClassicGraduate, sendNeiroGraduate } from "../API/services/graduate_services";
import { sendClassicInference, sendNeiroInference } from "../API/services/inference_services";
import { sendSeasonAnalytic } from "../API/services/season_analytic_services";
import LogStreamComponent from "../API/apiLogStreamComponent";


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
    const [responseText, setResponseText] = useState(""); // State for response message
    const [files, setFiles] = useState([]); // State for storing selected files
        
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
        try {
            const parsedRequest = JSON.parse(request);

            if (parsedRequest.inference) {
                if (parsedRequest.inference.dictmodels && (parsedRequest.inference.dictmodels.IFFT || parsedRequest.inference.dictmodels.IF)) {
                    console.log("Sending Neiro Inference request...");
                    const response = await sendNeiroInference(parsedRequest);
                    setResponseText(JSON.stringify(response, null, 2));
                } else {
                    console.log("Sending Classic Inference request...");
                    const response = await sendClassicInference(parsedRequest);
                    setResponseText(JSON.stringify(response, null, 2));
                }
            } else if (parsedRequest.dataset && parsedRequest.graduate) {
                if (parsedRequest.models_params) {
                    console.log("Sending Classic Graduate request...");
                    const response = await sendClassicGraduate(parsedRequest);
                    setResponseText(JSON.stringify(response, null, 2));
                } else {
                    console.log("Sending Neiro Graduate request...");
                    const response = await sendNeiroGraduate(parsedRequest);
                    setResponseText(JSON.stringify(response, null, 2));
                }
            } else if (parsedRequest.proccess) {
                console.log("Sending Season Analytic request...");
                const response = await sendSeasonAnalytic(parsedRequest);
                setResponseText(JSON.stringify(response, null, 2));
            } else {
                setResponseText("Invalid Query structure. Please check the format.");
            }
        } catch (error) {
            console.error("Failed to send query:", error);
            setResponseText("Error: Invalid JSON or API request failed.");
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
            console.error("Download error:", error);  // Log detailed error information
            setResponseText("Error while downloading the ZIP file."); // Update the error message for download
        }
    };

    // Функция, которая обновляет значение состояния запроса с новыми логами
    const handleNewLog = (newLog) => {
        setResponseText((prevRequest) => prevRequest + '\n' + newLog);  // Добавляем новый лог к существующему тексту
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
                    <Box
                        border="2px solid #FF0032"
                        borderRadius="10px"
                        p={5}
                        w="50%"
                        minH="600px"
                    >
                        <Text fontSize="24px" fontWeight="bold" color="#FFFFFF" mb={4}>
                            Insert Query
                        </Text>
                        <Textarea
                            value={request}
                            onChange={(e) => setRequest(e.target.value)}
                            height="500px"
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
                        flexDirection="column"  // Ensure proper flex alignment
                        alignItems="stretch"    // Ensure elements stretch vertically
                        justifyContent="flex-start"  // Align content to top
                        bg="transparent"
                    >
                        <Text fontSize="24px" fontWeight="bold" color="#FFFFFF" mb={4}>
                            LogStream
                        </Text>

                        {/* Компонент LogStreamComponent для получения новых логов */}
                        <LogStreamComponent onNewLog={handleNewLog} />

                        {/* Ensure Textarea uses all available height */}
                        <Textarea
                            value={responseText}  // Теперь мы используем состояние request
                            onChange={(e) => setResponseText(e.target.value)}  // Обновляем состояние при изменении
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
                        <label htmlFor="file-upload" style={{
                            fontSize: "Inter", fontWeight: "0", width: "200px", height: "44px",
                            backgroundColor: "#FF0032", color: "#FFFFFF", borderRadius: "10px",
                            display: "inline-block", textAlign: "center", lineHeight: "44px",
                            cursor: "pointer",
                            transition: "background-color 0.2s, transform 0.1s"
                        }}>
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
                            { label: "Load Zip", path: "#" }
                        ]}
                        showTitle={false}
                        hideButtons={false}
                        buttonWidth="200px"
                        buttonHeight="44px"
                        onClickActions={{
                            "Send CSV": handleSendCSV,     // Обработчик для кнопки "Send CSV"
                            "Send Query": handleSendQuery, // Обработчик для кнопки "Send Query"
                            "Load Zip": handleDownloadArchive // Обработчик для кнопки "Load Zip"
                        }}
                    />
                </HStack>
            </VStack>
        </Flex>
    );
};

export default QueryPage;
