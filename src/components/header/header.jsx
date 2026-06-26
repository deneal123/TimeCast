import React, { useEffect, useState } from "react";
import logo from "./../../images/logo.svg";
import {
  Flex, Image, Text, HStack, Button, Box, Tooltip, VStack,
  IconButton, useDisclosure, useBreakpointValue,
  Drawer, DrawerOverlay, DrawerContent, DrawerCloseButton, DrawerBody,
} from "@chakra-ui/react";
import { useNavigate, useLocation } from "react-router-dom";
import { HamburgerIcon } from "@chakra-ui/icons";
import { baseUrl } from "../../API/apiConsts";

const NAV_LINKS = [
  { label: "Главная",      path: "/main" },
  { label: "Дашборд",      path: "/query" },
  { label: "Задачи",       path: "/tasks" },
  { label: "Документация", path: "/documentation" },
];

const useBackendHealth = () => {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    const check = async () => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 4000);
      try {
        const r = await fetch(`${baseUrl}/tasks/`, { signal: controller.signal });
        setStatus(r.ok || r.status < 500);
      } catch {
        setStatus(false);
      } finally {
        clearTimeout(timer);
      }
    };

    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  return status;
};

const HealthDot = () => {
  const ok = useBackendHealth();
  const color = ok === null ? "#444" : ok ? "#48BB78" : "#FF0032";
  const label = ok === null ? "Проверка соединения…" : ok ? "Backend: online" : "Backend: offline";

  return (
    <Tooltip label={label} placement="bottom" hasArrow>
      <Box
        w="8px" h="8px"
        borderRadius="full"
        bg={color}
        flexShrink={0}
        cursor="default"
        sx={ok ? {
          boxShadow: `0 0 6px ${color}88`,
          animation: "pulse 2.5s ease-in-out infinite",
          "@keyframes pulse": {
            "0%, 100%": { opacity: 1 },
            "50%":       { opacity: 0.5 },
          },
        } : {}}
      />
    </Tooltip>
  );
};

const NavLink = ({ label, path, isActive, onClick }) => (
  <Button
    variant="ghost"
    color={isActive ? "#FFBF00" : "#888888"}
    fontWeight={isActive ? "600" : "400"}
    borderBottom={isActive ? "2px solid #FFBF00" : "2px solid transparent"}
    borderRadius={0}
    px={4}
    h="64px"
    _hover={{ color: "#FFFFFF", bg: "transparent" }}
    transition="all 0.15s"
    onClick={onClick}
    size="sm"
  >
    {label}
  </Button>
);

const Header = ({ showMenu = false }) => {
  const navigate   = useNavigate();
  const location   = useLocation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const isMobile   = useBreakpointValue({ base: true, md: false }) ?? false;

  const handleNav = (path) => {
    navigate(path);
    onClose();
  };

  return (
    <Flex
      as="header"
      position="sticky"
      top={0}
      zIndex={100}
      w="100%"
      h="64px"
      bg="#1A1D21"
      borderBottom="1px solid #FFFFFF0F"
      align="center"
      justify="space-between"
      px={[4, 6, 10]}
      backdropFilter="blur(12px)"
      flexShrink={0}
    >
      {/* Logo */}
      <Flex
        align="center"
        gap="10px"
        cursor="pointer"
        onClick={() => navigate("/main")}
        _hover={{ opacity: 0.85 }}
        transition="opacity 0.2s"
      >
        <Image src={logo} boxSize="32px" alt="Logo" />
        <Text
          fontFamily="Inter"
          fontWeight="700"
          fontSize="18px"
          bgGradient="linear(to-r, #FF0032, #FFBF00)"
          bgClip="text"
          color="transparent"
          letterSpacing="-0.02em"
        >
          TimeCast
        </Text>
      </Flex>

      {/* Desktop navigation */}
      {showMenu && !isMobile && (
        <HStack spacing={2}>
          {NAV_LINKS.map(({ label, path }) => (
            <NavLink
              key={path}
              label={label}
              path={path}
              isActive={location.pathname === path}
              onClick={() => navigate(path)}
            />
          ))}
          <HealthDot />
        </HStack>
      )}

      {/* Mobile hamburger */}
      {showMenu && isMobile && (
        <HStack spacing={3}>
          <HealthDot />
          <IconButton
            icon={<HamburgerIcon />}
            variant="ghost"
            color="#888"
            _hover={{ color: "#FFFFFF", bg: "transparent" }}
            onClick={onOpen}
            aria-label="Открыть меню"
            size="md"
          />
        </HStack>
      )}

      {/* Mobile drawer */}
      <Drawer isOpen={isOpen} onClose={onClose} placement="right" size="xs">
        <DrawerOverlay bg="blackAlpha.700" backdropFilter="blur(4px)" />
        <DrawerContent bg="#1A1D21" borderLeft="1px solid #FFFFFF0F">
          <DrawerCloseButton color="#888" top={4} right={4} />
          <DrawerBody pt={16} pb={6} px={5}>
            <VStack align="stretch" spacing={1}>
              {NAV_LINKS.map(({ label, path }) => {
                const isActive = location.pathname === path;
                return (
                  <Button
                    key={path}
                    variant="ghost"
                    justifyContent="flex-start"
                    color={isActive ? "#FFBF00" : "#888"}
                    fontWeight={isActive ? "600" : "400"}
                    bg={isActive ? "#FFBF0010" : "transparent"}
                    _hover={{ color: "#FFFFFF", bg: "#FFFFFF08" }}
                    borderRadius="10px"
                    h="44px"
                    px={4}
                    onClick={() => handleNav(path)}
                  >
                    {label}
                  </Button>
                );
              })}
            </VStack>

            <Box mt={8} pt={4} borderTop="1px solid #FFFFFF0F">
              <HStack spacing={2}>
                <HealthDot />
                <Text color="#444" fontSize="12px">Backend статус</Text>
              </HStack>
            </Box>
          </DrawerBody>
        </DrawerContent>
      </Drawer>
    </Flex>
  );
};

export default Header;
