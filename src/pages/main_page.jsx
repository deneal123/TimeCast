import React from "react";
import {
  Box,
  Flex,
  Heading,
  Text,
  Button,
  SimpleGrid,
  VStack,
  HStack,
  Icon,
  Badge,
} from "@chakra-ui/react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  TimeIcon,
  StarIcon,
  RepeatClockIcon,
  CalendarIcon,
  CheckCircleIcon,
  ArrowForwardIcon,
} from "@chakra-ui/icons";

const MotionBox  = motion(Box);
const MotionFlex = motion(Flex);

const FEATURES = [
  {
    icon: StarIcon,
    color: "#FF0032",
    title: "Классические модели",
    desc: "AUTOARIMA, AUTOREG, AUTOETS, PROPHET, TBATS — автоматический подбор лучшей модели на каждый ряд.",
    badge: "Classic",
    badgeColor: "red",
  },
  {
    icon: TimeIcon,
    color: "#FFBF00",
    title: "Нейросети (iTransformer)",
    desc: "Transformer-архитектура IF / IFFT с reversible instance norm — для сложных многомерных паттернов.",
    badge: "Neural",
    badgeColor: "yellow",
  },
  {
    icon: RepeatClockIcon,
    color: "#00B5D8",
    title: "Фоновое обучение",
    desc: "Поставьте задачу в очередь и следите за статусом в реальном времени — сервер обучает без блокировки UI.",
    badge: "Queue",
    badgeColor: "cyan",
  },
  {
    icon: CalendarIcon,
    color: "#9F7AEA",
    title: "Сезонная декомпозиция",
    desc: "Разложение ряда на тренд, сезонность и остаток с интерактивным Plotly-графиком.",
    badge: "Analytics",
    badgeColor: "purple",
  },
];

const STATS = [
  { value: "5+", label: "Classic-моделей" },
  { value: "2",  label: "Neural-архитектуры" },
  { value: "∞",  label: "Временных рядов" },
  { value: "REST", label: "API + SSE-логи" },
];

const STEPS = [
  {
    num: "01",
    color: "#FF0032",
    title: "Загрузите данные",
    desc: "Передайте train / test / features CSV через дашборд или REST API.",
  },
  {
    num: "02",
    color: "#FFBF00",
    title: "Настройте запрос",
    desc: "Конструктор или JSON напрямую — выберите модель, периоды, режим.",
  },
  {
    num: "03",
    color: "#48BB78",
    title: "Получите результат",
    desc: "Граф прогноза, метрики R² и RMSE — синхронно или в фоне.",
  },
];

const FeatureCard = ({ icon, color, title, desc, badge, badgeColor, index }) => (
  <MotionBox
    initial={{ opacity: 0, y: 24 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.3 + index * 0.1, duration: 0.5 }}
    whileHover={{ scale: 1.03, translateY: -4 }}
    bg="#141820"
    border="1px solid"
    borderColor={`${color}44`}
    borderRadius="16px"
    p={6}
    cursor="default"
    _hover={{ borderColor: color, boxShadow: `0 0 24px ${color}22` }}
    sx={{ transition: "all 0.25s ease" }}
  >
    <HStack mb={3} justify="space-between">
      <Box
        w="40px" h="40px" borderRadius="10px"
        bg={`${color}1A`}
        display="flex" alignItems="center" justifyContent="center"
      >
        <Icon as={icon} color={color} boxSize="20px" />
      </Box>
      <Badge colorScheme={badgeColor} fontSize="10px" px={2} borderRadius="6px">
        {badge}
      </Badge>
    </HStack>
    <Text color="#FFFFFF" fontWeight="600" fontSize="16px" mb={2}>{title}</Text>
    <Text color="#888" fontSize="14px" lineHeight="1.6">{desc}</Text>
  </MotionBox>
);

const MainPage = () => {
  const navigate = useNavigate();

  return (
    <Box
      minH="100vh" w="100%"
      bg="#0A050D"
      bgGradient="linear(to-br, #0A050D, #0D0A1A, #0A050D)"
      overflowX="hidden"
    >
      {/* Hero */}
      <Flex
        direction="column" align="center" justify="center"
        minH="72vh" px={8} pt={20} pb={10}
        position="relative"
      >
        <Box
          position="absolute" top="30%" left="50%"
          transform="translate(-50%, -50%)"
          w="600px" h="300px"
          bg="radial-gradient(ellipse, #FF003222 0%, transparent 70%)"
          pointerEvents="none" zIndex={0}
        />

        <MotionFlex
          direction="column" align="center" zIndex={1}
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          <Badge
            colorScheme="red" fontSize="11px" px={3} py={1}
            borderRadius="full" mb={5} letterSpacing="0.1em"
          >
            TIME SERIES PLATFORM
          </Badge>

          <Heading
            fontSize={["36px", "48px", "60px", "72px"]}
            fontWeight="800" textAlign="center" lineHeight="1.1" mb={5}
            bgGradient="linear(to-r, #FF0032, #FFBF00)"
            bgClip="text" color="transparent"
          >
            TimeCast
          </Heading>

          <Text
            color="#AAAAAA" fontSize={["16px", "18px", "20px"]}
            textAlign="center" maxW="600px" lineHeight="1.7" mb={8}
          >
            Прогнозирование временных рядов с классическими и нейросетевыми моделями.
            REST API, фоновое обучение, интерактивная визуализация.
          </Text>

          <HStack spacing={4}>
            <Button
              size="lg" bg="#FF0032" color="#FFFFFF"
              _hover={{ bg: "#D0021B", transform: "translateY(-2px)" }}
              _active={{ transform: "translateY(0)" }}
              transition="all 0.2s" borderRadius="10px" px={8} fontWeight="600"
              rightIcon={<Icon as={ArrowForwardIcon} />}
              onClick={() => navigate("/query")}
            >
              Открыть дашборд
            </Button>
            <Button
              size="lg" variant="outline" borderColor="#FFBF00" color="#FFBF00"
              _hover={{ bg: "#FFBF0011", transform: "translateY(-2px)" }}
              _active={{ transform: "translateY(0)" }}
              transition="all 0.2s" borderRadius="10px" px={8} fontWeight="600"
              onClick={() => navigate("/documentation")}
            >
              Документация
            </Button>
          </HStack>
        </MotionFlex>
      </Flex>

      {/* Stats bar */}
      <MotionBox
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        transition={{ delay: 0.4, duration: 0.6 }}
        borderTop="1px solid #FFFFFF0F" borderBottom="1px solid #FFFFFF0F"
        py={5} px={8} mb={14}
      >
        <SimpleGrid columns={[2, 4]} maxW="700px" mx="auto" spacing={6}>
          {STATS.map(({ value, label }) => (
            <VStack key={label} spacing={0}>
              <Text color="#FF0032" fontSize={["24px", "28px"]} fontWeight="700" fontFamily="Inter">
                {value}
              </Text>
              <Text color="#666" fontSize="13px">{label}</Text>
            </VStack>
          ))}
        </SimpleGrid>
      </MotionBox>

      {/* How it works */}
      <MotionBox
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.6 }}
        px={[4, 8, 12]} pb={14} maxW="1000px" mx="auto"
      >
        <Text
          color="#555" fontSize="12px" fontWeight="600" letterSpacing="0.12em"
          textAlign="center" mb={8} textTransform="uppercase"
        >
          Как это работает
        </Text>

        <Flex direction={["column", "column", "row"]} gap={[4, 4, 0]} align={["stretch", "stretch", "stretch"]}>
          {STEPS.map((step, i) => (
            <React.Fragment key={step.num}>
              <Box
                flex={1}
                bg="#141820"
                border="1px solid"
                borderColor={`${step.color}33`}
                borderRadius="14px"
                p={6}
                _hover={{ borderColor: step.color, boxShadow: `0 0 20px ${step.color}18` }}
                transition="all 0.2s"
              >
                <Text
                  color={step.color} fontSize="32px" fontWeight="800"
                  fontFamily="monospace" mb={3} lineHeight={1} opacity={0.9}
                >
                  {step.num}
                </Text>
                <Text color="#FFFFFF" fontWeight="600" fontSize="15px" mb={2}>
                  {step.title}
                </Text>
                <Text color="#777" fontSize="13px" lineHeight="1.65">
                  {step.desc}
                </Text>
              </Box>
              {i < STEPS.length - 1 && (
                <Flex
                  align="center" justify="center"
                  px={[0, 0, 3]}
                  py={[2, 2, 0]}
                  display={["none", "none", "flex"]}
                >
                  <Icon as={ArrowForwardIcon} color="#333" boxSize="20px" />
                </Flex>
              )}
            </React.Fragment>
          ))}
        </Flex>
      </MotionBox>

      {/* Feature cards */}
      <Box px={[4, 8, 12, 16]} pb={20} maxW="1200px" mx="auto">
        <Text
          color="#555" fontSize="12px" fontWeight="600" letterSpacing="0.12em"
          textAlign="center" mb={8} textTransform="uppercase"
        >
          Возможности
        </Text>
        <SimpleGrid columns={[1, 2, 2, 4]} spacing={5}>
          {FEATURES.map((f, i) => (
            <FeatureCard key={f.title} {...f} index={i} />
          ))}
        </SimpleGrid>

        {/* Quick start hint */}
        <MotionBox
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }}
          mt={10} p={5} bg="#141820" border="1px solid #FFFFFF0A"
          borderRadius="14px" maxW="700px" mx="auto"
        >
          <HStack spacing={3} mb={2}>
            <Icon as={CheckCircleIcon} color="#48BB78" boxSize="16px" />
            <Text color="#FFFFFF" fontWeight="600" fontSize="14px">Быстрый старт</Text>
          </HStack>
          <Text color="#777" fontSize="13px" lineHeight="1.7">
            Перейдите в{" "}
            <Text as="span" color="#FFBF00" cursor="pointer"
              _hover={{ textDecoration: "underline" }} onClick={() => navigate("/query")}>
              дашборд
            </Text>
            , выберите retail-шаблон (или постройте запрос через форму), нажмите{" "}
            <Text as="span" color="#FF0032">Send Query</Text>{" "}
            — и через несколько секунд получите интерактивный прогноз с метриками.
          </Text>
        </MotionBox>
      </Box>
    </Box>
  );
};

export default MainPage;
