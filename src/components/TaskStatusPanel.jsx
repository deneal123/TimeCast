import React, { useEffect, useRef, useState } from "react";
import { Box, Badge, Text, HStack, Divider } from "@chakra-ui/react";
import { getTask } from "../API/services/task_services";
import ForecastChart from "./ForecastChart";
import TrainingResults from "./TrainingResults";

const STATUS_COLOR = {
  pending: "gray",
  running: "yellow",
  done: "green",
  failed: "red",
};

const POLL_INTERVAL_MS = 2500;

const TaskStatusPanel = ({ taskId, onResultReady }) => {
  const [record, setRecord] = useState(null);
  const activeRef = useRef(true);

  useEffect(() => {
    if (!taskId) return;
    activeRef.current = true;

    const poll = async () => {
      while (activeRef.current) {
        try {
          const data = await getTask(taskId);
          setRecord(data);
          if (data.status === "done" && data.result) onResultReady?.(data.result);
          if (data.status === "done" || data.status === "failed") break;
        } catch (e) {
          console.error("Task poll error:", e);
        }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      }
    };

    poll();
    return () => { activeRef.current = false; };
  }, [taskId, onResultReady]);

  if (!record) return null;

  const isTraining = record.result && "best_model" in (Object.values(Object.values(record.result)[0] ?? {})[0] ?? {});

  return (
    <Box border="2px solid #FF0032" borderRadius="10px" p={5} bg="#1A1A1A" mt={4} w="100%">
      <HStack mb={3} justify="space-between">
        <Text color="#FFFFFF" fontWeight="bold" fontSize="18px">
          Задача {record.task_id}
        </Text>
        <HStack spacing={3}>
          <Badge colorScheme={STATUS_COLOR[record.status] ?? "gray"} fontSize="13px" px={2}>
            {record.status}
          </Badge>
          <Text color="#888" fontSize="12px">
            {record.operation}
          </Text>
        </HStack>
      </HStack>

      {record.status === "running" && (
        <Text color="#FFBF00" fontSize="13px">Обучение выполняется…</Text>
      )}
      {record.status === "failed" && (
        <Text color="#FF0032" fontSize="13px">{record.error}</Text>
      )}

      {record.status === "done" && record.result && (
        <>
          <Divider my={3} borderColor="#444" />
          {isTraining ? (
            <TrainingResults results={record.result.results ?? record.result} />
          ) : (
            <ForecastChart results={record.result.results ?? record.result} />
          )}
        </>
      )}
    </Box>
  );
};

export default TaskStatusPanel;
