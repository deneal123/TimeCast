import React from "react";
import { Flex, Spinner } from "@chakra-ui/react";
import { AnimatePresence, motion } from "framer-motion";
import Header from "./components/header/header";
import { Outlet, useLocation } from "react-router-dom";
import Footer from "./components/footer";

const MotionFlex = motion(Flex);

const PAGE_TRANSITION = {
  initial:    { opacity: 0, y: 10 },
  animate:    { opacity: 1, y: 0, transition: { duration: 0.22, ease: "easeOut" } },
  exit:       { opacity: 0, y: -6, transition: { duration: 0.15, ease: "easeIn" } },
};

const PageLoader = () => (
  <Flex align="center" justify="center" flex={1} minH="200px">
    <Spinner color="#FF0032" size="lg" thickness="3px" speed="0.6s" />
  </Flex>
);

function Layout() {
  const location = useLocation();
  const isMain = location.pathname === "/main";

  return (
    <Flex direction="column" minH="100vh" w="100%" bg="menu_mts">
      <Header showMenu={!isMain} />
      <React.Suspense fallback={<PageLoader />}>
        <AnimatePresence mode="wait" initial={false}>
          <MotionFlex
            key={location.pathname}
            direction="column"
            flex={1}
            variants={PAGE_TRANSITION}
            initial="initial"
            animate="animate"
            exit="exit"
          >
            <Outlet />
          </MotionFlex>
        </AnimatePresence>
      </React.Suspense>
      {!isMain && <Footer />}
    </Flex>
  );
}

export default Layout;
