import React from "react";
import { Box, Text, Button, VStack } from "@chakra-ui/react";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error("ErrorBoundary caught:", error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box p={8} textAlign="center">
          <VStack spacing={4}>
            <Text fontSize="24px" color="#FF0032" fontWeight="bold">
              Что-то пошло не так
            </Text>
            <Text color="#AAA" maxW="480px">
              {this.state.error?.message || "Неизвестная ошибка"}
            </Text>
            <Button
              onClick={() => this.setState({ hasError: false, error: null })}
              bg="#FF0032"
              color="#FFFFFF"
              _hover={{ bg: "#D0021B" }}
              borderRadius="10px"
            >
              Попробовать снова
            </Button>
          </VStack>
        </Box>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
