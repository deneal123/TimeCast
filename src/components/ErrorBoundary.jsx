import React from "react";
import { Box, Text, Button, Flex, Icon } from "@chakra-ui/react";
import { WarningTwoIcon } from "@chakra-ui/icons";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error("ErrorBoundary:", error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Flex
          direction="column"
          align="center"
          justify="center"
          minH="100vh"
          bg="#0F1218"
          px={6}
        >
          <Box
            bg="#141820"
            border="1px solid #FF003444"
            borderRadius="16px"
            p={8}
            maxW="480px"
            w="100%"
            textAlign="center"
          >
            <Icon as={WarningTwoIcon} color="#FF0032" boxSize="36px" mb={4} />
            <Text color="#FFFFFF" fontSize="20px" fontWeight="700" mb={2}>
              Что-то пошло не так
            </Text>
            <Text color="#666" fontSize="14px" mb={6} lineHeight="1.7">
              {this.state.error?.message || "Неизвестная ошибка компонента"}
            </Text>
            <Button
              bg="#FF0032"
              color="#FFFFFF"
              _hover={{ bg: "#D0021B", transform: "translateY(-1px)" }}
              _active={{ transform: "translateY(0)" }}
              transition="all 0.15s"
              borderRadius="10px"
              px={8}
              onClick={() => this.setState({ hasError: false, error: null })}
            >
              Попробовать снова
            </Button>
          </Box>
        </Flex>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
