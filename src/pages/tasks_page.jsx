import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  Box,
  Text,
  HStack,
  Badge,
  Flex,
  Button,
  Icon,
  Spinner,
  Divider,
  Collapse,
  useDisclosure,
} from "@chakra-ui/react";
import {
  RepeatIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  CheckCircleIcon,
  WarningIcon,
  TimeIcon,
} from "@chakra-ui/icons";
import { listTasks } from "../API/services/task_services";
import TaskStatusPanel from "../components/TaskStatusPanel";

const STATUS_META = {
  pending: { color: "#888",    scheme: "gray",   icon: TimeIcon,        label: "В очереди",    dot: "#888" },
  running: { color: "#FFBF00", scheme: "yellow",  icon: null,            label: "Выполняется",  dot: "#FFBF00" },
  done:    { color: "#48BB78", scheme: "green",   icon: CheckCircleIcon, label: "Готово",       dot: "#48BB78" },
  failed:  { color: "#FF8888", scheme: "red",     icon: WarningIcon,     label: "Ошибка",       dot: "#FF8888" },
};

const relativeTime = (ts) => {
  if (!ts) return "";
  const diff = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
  if (diff < 60) return `${diff}с назад`;
  if (diff < 3600) return `${Math.floor(diff / 60)}м назад`;
  return `${Math.floor(diff / 3600)}ч назад`;
};

const TaskRow = ({ task }) => {
  const { isOpen, onToggle } = useDisclosure();
  const meta = STATUS_META[task.status] ?? STATUS_META.pending;

  return (
    <Box>
      <Flex
        px={5}
        py={3}
        align="center"
        justify="space-between"
        cursor="pointer"
        _hover={{ bg: "#0F1218" }}
        onClick={onToggle}
        transition="background 0.15s"
      >
        <HStack spacing={3} flex={1} minW={0}>
          {/* Status dot / spinner */}
          {task.status === "running" ? (
            <Spinner size="xs" color="#FFBF00" thickness="2px" flexShrink={0} />
          ) : (
            <Box
              w="8px"
              h="8px"
              borderRadius="full"
              bg={meta.dot}
              flexShrink={0}
            />
          )}

          {/* Operation badge */}
          <Badge
            colorScheme={meta.scheme}
            fontSize="10px"
            px={2}
            borderRadius="6px"
            flexShrink={0}
          >
            {task.operation ?? "—"}
          </Badge>

          {/* Task ID */}
          <Text
            color="#555"
            fontSize="12px"
            fontFamily="monospace"
            isTruncated
            maxW={["100px", "160px", "240px"]}
          >
            {task.task_id}
          </Text>
        </HStack>

        <HStack spacing={3} flexShrink={0}>
          <Badge colorScheme={meta.scheme} fontSize="10px" px={2} borderRadius="6px">
            {meta.label}
          </Badge>
          <Text color="#444" fontSize="11px" minW="60px" textAlign="right">
            {relativeTime(task.created_at)}
          </Text>
          <Icon
            as={isOpen ? ChevronUpIcon : ChevronDownIcon}
            color="#555"
            boxSize="14px"
          />
        </HStack>
      </Flex>

      <Collapse in={isOpen} animateOpacity>
        <Box px={5} pb={4}>
          <TaskStatusPanel taskId={task.task_id} />
        </Box>
      </Collapse>
    </Box>
  );
};

const CountChip = ({ label, count, color }) => (
  <HStack spacing={1}>
    <Box w="7px" h="7px" borderRadius="full" bg={color} />
    <Text color="#555" fontSize="12px">
      {label}:
    </Text>
    <Text color={count > 0 ? color : "#333"} fontSize="12px" fontWeight="600">
      {count}
    </Text>
  </HStack>
);

const TasksPage = () => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState(null);
  const timerRef = useRef(null);

  const hasActive = tasks.some((t) => t.status === "pending" || t.status === "running");

  const fetchTasks = useCallback(async () => {
    try {
      const data = await listTasks();
      const list = Array.isArray(data) ? data : (data.tasks ?? []);
      list.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      setTasks(list);
      setLastRefresh(new Date());
    } catch {
      // ignore — will retry on next poll
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);

  // Poll faster when there are active tasks
  useEffect(() => {
    const interval = hasActive ? 3000 : 10000;
    timerRef.current = setTimeout(() => {
      fetchTasks();
    }, interval);
    return () => clearTimeout(timerRef.current);
  }, [tasks, hasActive, fetchTasks]);

  const counts = {
    pending: tasks.filter((t) => t.status === "pending").length,
    running: tasks.filter((t) => t.status === "running").length,
    done:    tasks.filter((t) => t.status === "done").length,
    failed:  tasks.filter((t) => t.status === "failed").length,
  };

  return (
    <Flex direction="column" w="100%" px={[4, 6, 8]} pt={6} pb={10} align="center" flexGrow={1}>
      <Box w="100%" maxW="1000px">

        {/* Page header */}
        <HStack justify="space-between" mb={5}>
          <Box>
            <Text color="#FFFFFF" fontWeight="700" fontSize="22px" letterSpacing="-0.02em">
              Очередь задач
            </Text>
            {lastRefresh && (
              <Text color="#444" fontSize="12px" mt={1}>
                Обновлено: {lastRefresh.toLocaleTimeString()}
              </Text>
            )}
          </Box>
          <Button
            size="sm"
            variant="outline"
            borderColor="#2A2E36"
            color="#888"
            _hover={{ borderColor: "#555", color: "#FFFFFF" }}
            leftIcon={<Icon as={RepeatIcon} />}
            borderRadius="8px"
            onClick={fetchTasks}
            isLoading={loading}
          >
            Обновить
          </Button>
        </HStack>

        {/* Stats row */}
        <HStack
          spacing={6}
          mb={5}
          px={5}
          py={3}
          bg="#0F1218"
          border="1px solid #2A2E36"
          borderRadius="12px"
        >
          <CountChip label="В очереди"  count={counts.pending} color="#888" />
          <CountChip label="Выполняется" count={counts.running} color="#FFBF00" />
          <CountChip label="Готово"     count={counts.done}    color="#48BB78" />
          <CountChip label="Ошибка"     count={counts.failed}  color="#FF8888" />
        </HStack>

        {/* Task list */}
        <Box
          border="1px solid #2A2E36"
          borderRadius="14px"
          overflow="hidden"
          bg="#141820"
        >
          {/* Header */}
          <HStack
            px={5}
            py={3}
            bg="#0F1218"
            borderBottom="1px solid #2A2E36"
            spacing={3}
          >
            <Text color="#FFFFFF" fontWeight="600" fontSize="15px">
              Задачи
            </Text>
            {tasks.length > 0 && (
              <Badge colorScheme="gray" fontSize="10px" px={2} borderRadius="6px">
                {tasks.length}
              </Badge>
            )}
            {hasActive && (
              <HStack spacing={1} ml={1}>
                <Spinner size="xs" color="#FFBF00" thickness="2px" />
                <Text color="#FFBF00" fontSize="11px">активные</Text>
              </HStack>
            )}
          </HStack>

          {/* Rows */}
          {loading && tasks.length === 0 ? (
            <Flex align="center" justify="center" py={12}>
              <Spinner color="#555" size="md" thickness="2px" />
            </Flex>
          ) : tasks.length === 0 ? (
            <Flex direction="column" align="center" justify="center" py={16} gap={3}>
              <Icon as={TimeIcon} color="#2A2E36" boxSize="32px" />
              <Text color="#444" fontSize="14px">Нет задач в очереди</Text>
              <Text color="#333" fontSize="12px">
                Поставьте задачу обучения из Дашборда (Ctrl+Shift+Enter)
              </Text>
            </Flex>
          ) : (
            tasks.map((task, i) => (
              <Box key={task.task_id}>
                <TaskRow task={task} />
                {i < tasks.length - 1 && <Divider borderColor="#1A1D21" />}
              </Box>
            ))
          )}
        </Box>

      </Box>
    </Flex>
  );
};

export default TasksPage;
