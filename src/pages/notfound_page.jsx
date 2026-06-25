import { Flex, Heading, Text, Button } from "@chakra-ui/react";
import { useNavigate } from "react-router-dom";

const NotFoundPage = () => {
  const navigate = useNavigate();
  return (
    <Flex direction="column" align="center" justify="center" minH="80vh" gap={5}>
      <Heading
        fontSize="96px"
        fontWeight="800"
        bgGradient="linear(to-r, #FF0032, #FFBF00)"
        bgClip="text"
        color="transparent"
        lineHeight="1"
      >
        404
      </Heading>
      <Text color="#666" fontSize="18px">
        Страница не найдена
      </Text>
      <Button
        size="md"
        bg="#FF0032"
        color="#FFFFFF"
        _hover={{ bg: "#D0021B" }}
        borderRadius="10px"
        px={8}
        onClick={() => navigate("/main")}
      >
        На главную
      </Button>
    </Flex>
  );
};

export default NotFoundPage;
