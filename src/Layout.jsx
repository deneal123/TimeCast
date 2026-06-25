import React from "react";
import { Flex } from "@chakra-ui/react";
import Header from "./components/header/header";
import { Outlet, useLocation } from "react-router-dom";
import Footer from "./components/footer";

const NAV = [
  { label: "Главная", path: "/main" },
  { label: "Дашборд", path: "/query" },
  { label: "Документация", path: "/documentation" },
];

function Layout() {
  const location = useLocation();
  const isMain = location.pathname === "/main";

  const menuButtons = isMain
    ? []
    : NAV.filter((n) => n.path !== location.pathname);

  return (
    <Flex direction="column" minH="100vh" w="100%" bg="menu_mts">
      <Header
        showMenu={!isMain}
        hideButtons={isMain}
        menuButtons={menuButtons}
      />
      <Flex direction="column" flex={1}>
        <Outlet />
      </Flex>
      {!isMain && <Footer />}
    </Flex>
  );
}

export default Layout;
