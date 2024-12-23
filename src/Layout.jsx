import React from "react";
import { VStack } from "@chakra-ui/react";
import Header from "./components/header/header";
import { Outlet, useLocation } from "react-router-dom";
import Footer from "./components/footer";

function Layout() {
    const location = useLocation();

    // ������ ��� �����������, ����� ������ �������� � Header
    let showMenu = true;
    let hideButtons = false;
    let menuButtons = [];

    switch (location.pathname) {
        case "/main":
            showMenu = false;
            hideButtons = true; // �������� ������ �� ������� ��������
            break;
        case "/documentation":
            menuButtons = [
                { label: "Main", path: "/" },
                { label: "Query", path: "/query" },
            ];
            hideButtons = false;
            break;
        case "/query":
            menuButtons = [
                { label: "Main", path: "/" },
                { label: "Documentation", path: "/documentation" },
            ];
            hideButtons = false;
            break;
        default:
            // �� ����� ����������� �������� (��������, 404) ����� ���������� ����� ������
            menuButtons = [
                { label: "Main", path: "/" },
                { label: "Documentation", path: "/documentation" },
                { label: "Query", path: "/query" },
            ];
            hideButtons = false;
            break;
    }

    return (
        <VStack backgroundColor="menu_mts" width="100%" minH="100vh">
            <Header
                showMenu={showMenu}
                hideButtons={hideButtons}
                menuButtons={menuButtons} />
            <Outlet />
            <Footer />
        </VStack>
    );
}

export default Layout;
