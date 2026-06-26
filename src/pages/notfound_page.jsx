import { Flex, Heading, Text, Button } from "@chakra-ui/react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

const MotionFlex    = motion(Flex);
const MotionHeading = motion(Heading);
const MotionText    = motion(Text);
const MotionButton  = motion(Button);

const NotFoundPage = () => {
  const navigate = useNavigate();

  return (
    <MotionFlex
      direction="column"
      align="center"
      justify="center"
      minH="80vh"
      gap={5}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <MotionHeading
        fontSize="120px"
        fontWeight="800"
        bgGradient="linear(to-r, #FF0032, #FFBF00)"
        bgClip="text"
        color="transparent"
        lineHeight="1"
        initial={{ opacity: 0, scale: 0.8, y: -20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
      >
        404
      </MotionHeading>

      <MotionText
        color="#555"
        fontSize="18px"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15, duration: 0.4 }}
      >
        Страница не найдена
      </MotionText>

      <MotionText
        color="#333"
        fontSize="14px"
        textAlign="center"
        maxW="340px"
        lineHeight="1.6"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25, duration: 0.4 }}
      >
        Возможно, она была удалена или вы перешли по неверной ссылке.
      </MotionText>

      <MotionButton
        size="md"
        bg="#FF0032"
        color="#FFFFFF"
        _hover={{ bg: "#D0021B", transform: "translateY(-2px)" }}
        _active={{ transform: "translateY(0)" }}
        borderRadius="10px"
        px={8}
        transition="all 0.2s"
        onClick={() => navigate("/main")}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        // eslint-disable-next-line react/prop-types
        sx={{ transition: "all 0.2s ease" }}
      >
        На главную
      </MotionButton>
    </MotionFlex>
  );
};

export default NotFoundPage;
