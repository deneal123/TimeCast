import React from "react";
import logo from "./../../images/logo.svg";
import { Flex, Image, Text, HStack, Button } from "@chakra-ui/react";
import { useNavigate, useLocation } from "react-router-dom";

const NAV_LINKS = [
  { label: "Главная", path: "/main" },
  { label: "Дашборд", path: "/query" },
  { label: "Документация", path: "/documentation" },
];

const Header = ({ showMenu = false, hideButtons = false, menuButtons = [] }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const visibleLinks = menuButtons.length > 0
    ? NAV_LINKS.filter((l) => menuButtons.some((m) => m.path === l.path))
    : NAV_LINKS;

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
      {/* Logo + title */}
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

      {/* Navigation */}
      {showMenu && !hideButtons && (
        <HStack spacing={1}>
          {visibleLinks.map(({ label, path }) => {
            const isActive = location.pathname === path;
            return (
              <Button
                key={path}
                size="sm"
                variant="ghost"
                color={isActive ? "#FFBF00" : "#888888"}
                fontWeight={isActive ? "600" : "400"}
                borderBottom={isActive ? "2px solid #FFBF00" : "2px solid transparent"}
                borderRadius={0}
                px={4}
                h="64px"
                _hover={{ color: "#FFFFFF", bg: "transparent" }}
                transition="all 0.15s"
                onClick={() => navigate(path)}
              >
                {label}
              </Button>
            );
          })}
        </HStack>
      )}
    </Flex>
  );
};

export default Header;
