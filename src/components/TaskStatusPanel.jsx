import React, { useEffect, useRef, useState } from "react";
import {
  Box,
  Badge,
  Text,
  HStack,
  Divider,
  Link,
  Flex,
  Spinner,
  Icon,
} from "@chakra-ui/react";
import { ExternalLinkIcon, WarningIcon, CheckCircleIcon, TimeIcon } from "@chakra-ui/icons";
import { getTask } from "../API/services/task_services";
import ForecastChart from "./ForecastChart";
import TrainingResults from "./TrainingResults";

const STATUS_META = {
  pending: { color: "gray",   scheme: "gray",   icon: TimeIcon,        label: "В очереди" },
  running: { color: "#FFBF00", scheme: "yellow", icon: null,            label: "Выполняется" },
  done:    { color: "#48BB78", scheme: "green",  icon: CheckCircleIcon, label: "Готово" },
  failed:  { color: "#FF8888", scheme: "red",    icon: WarningIcon,     label: "Ошибка" },
};

const POLL_INTERVAL_MS = 2500;

const fmtElapsed = (s) => {
  if (s < 60) return `${s}с`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r > 0 ? `${m}м ${r}с` : `${m}м`;
};

const TaskStatusPanel = ({ taskId, onResultReady }) => {
  const [record,  setRecord]  = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const activeRef  = useRef(true);
  const timerRef   = useRef(null);

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

  // Elapsed timer while running
  useEffect(() => {
    if (record?.status === "running") {
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [record?.status]);

  if (!record) return null;

  const meta = STATUS_META[record.status] ?? STATUS_META.pending;
  const isTraining =
    record.result &&
    "best_model" in
      (Object.values(Object.values(Object.values(record.result)[0] ?? {})[0] ?? {})[0] ?? {});

  const artifacts = record.result?.artifacts ?? [];

  return (
    <Box
      bg="#141820"
      border="1px solid #2A2E36"
      borderRadius="14px"
      overflow="hidden"
      mt={5}
      w="100%"
    >
      {/* Header */}
      <HStack
        px={5}
        py={3}
        bg="#0F1218"
        borderBottom="1px solid #2A2E36"
        justify="space-between"
      >
        <HStack spacing={3}>
          <Text color="#FFFFFF" fontWeight="600" fontSize="15px">
            Задача
          </Text>
          <Text color="#555" fontSize="12px" fontFamily="monospace">
            {record.task_id}
          </Text>
        </HStack>
        <HStack spacing={3}>
          <Badge colorScheme={meta.scheme} fontSize="11px" px={2} borderRadius="6px">
            {record.operation}
          </Badge>
          <Badge colorScheme={meta.scheme} fontSize="11px" px={2} borderRadius="6px">
            {meta.label}
          </Badge>
        </HStack>
      </HStack>

      {/* Status body */}
      <Box px={5} py={4}>
        {record.status === "running" && (
          <HStack spacing={3}>
            <Spinner size="sm" color="#FFBF00" thickness="2px" />
            <Text fontSize="13px" color="#FFBF00">Обучение выполняется…</Text>
            {elapsed > 0 && (
              <Text fontSize="12px" color="#666" fontFamily="monospace">
                {fmtElapsed(elapsed)}
              </Text>
            )}
          </HStack>
        )}

        {record.status === "pending" && (
          <HStack spacing={3} color="#888">
            <Spinner size="sm" color="#888" thickness="2px" />
            <Text fontSize="13px">Ожидание в очереди…</Text>
          </HStack>
        )}

        {record.status === "failed" && (
          <Flex
            align="flex-start"
            gap={3}
            bg="#1A0A0A"
            border="1px solid #FF003244"
            borderRadius="10px"
            p={4}
          >
            <Icon as={WarningIcon} color="#FF8888" mt="2px" flexShrink={0} />
            <Text color="#FF8888" fontSize="13px" fontFamily="monospace">
              {record.error || "Неизвестная ошибка"}
            </Text>
          </Flex>
        )}

        {/* Timestamps */}
        {(record.created_at || record.updated_at) && (
          <HStack mt={3} spacing={4}>
            {record.created_at && (
              <Text color="#444" fontSize="11px">
                Создана: {new Date(record.created_at).toLocaleTimeString()}
              </Text>
            )}
            {record.updated_at && record.updated_at !== record.created_at && (
              <Text color="#444" fontSize="11px">
                Обновлена: {new Date(record.updated_at).toLocaleTimeString()}
              </Text>
            )}
          </HStack>
        )}

        {/* Artifacts */}
        {artifacts.length > 0 && (
          <>
            <Divider borderColor="#2A2E36" my={4} />
            <Text color="#555" fontSize="11px" fontWeight="600" letterSpacing="0.08em" mb={2}>
              АРТЕФАКТЫ S3 ({artifacts.length})
            </Text>
            <Flex direction="column" gap={1}>
              {artifacts.map((a) => (
                <HStack
                  key={a.key}
                  px={3}
                  py={2}
                  bg="#0D1017"
                  borderRadius="8px"
                  justify="space-between"
                >
                  <Link
                    href={`/server/tasks/${record.task_id}/artifacts/${encodeURIComponent(a.key)}`}
                    isExternal
                    color="#FFBF00"
                    fontSize="12px"
                    fontFamily="monospace"
                    _hover={{ color: "#FFD540" }}
                  >
                    {a.key.split("/").slice(-2).join("/")}
                    <Icon as={ExternalLinkIcon} ml={1} boxSize="10px" />
                  </Link>
                  <Text color="#444" fontSize="11px" flexShrink={0}>
                    {(a.size / 1024).toFixed(1)} KB
                  </Text>
                </HStack>
              ))}
            </Flex>
          </>
        )}
      </Box>

      {/* Result charts */}
      {record.status === "done" && record.result && (
        <>
          <Divider borderColor="#2A2E36" />
          <Box p={5}>
            {isTraining ? (
              <TrainingResults results={record.result.results ?? record.result} />
            ) : (
              <ForecastChart results={record.result.results ?? record.result} />
            )}
          </Box>
        </>
      )}
    </Box>
  );
};

export default TaskStatusPanel;
