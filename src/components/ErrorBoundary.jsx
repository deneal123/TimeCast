import React from "react";
import { Box, Flex, Text, Button, Icon, HStack, Code } from "@chakra-ui/react";
import { WarningTwoIcon, RepeatIcon } from "@chakra-ui/icons";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null, resetKey: 0 };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState((s) => ({ error: null, resetKey: s.resetKey + 1 }));
    window.location.hash = "#/main";
  };

  render() {
    if (!this.state.error) {
      return (
        <React.Fragment key={this.state.resetKey}>
          {this.props.children}
        </React.Fragment>
      );
    }

    const msg = this.state.error?.message || String(this.state.error);

    return (
      <Flex direction="column" align="center" justify="center" minH="60vh" px={6} gap={5}>
        <Icon as={WarningTwoIcon} color="#FF0032" boxSize="40px" />

        <Box textAlign="center">
          <Text color="#FFFFFF" fontWeight="700" fontSize="20px" mb={2}>
            Что-то пошло не так
          </Text>
          <Text color="#555" fontSize="14px" maxW="480px">
            Произошла ошибка при отображении страницы. Попробуйте перезагрузить
            или вернуться на главную.
          </Text>
        </Box>

        {msg && (
          <Box
            bg="#0D1017"
            border="1px solid #2A2E36"
            borderRadius="10px"
            px={4}
            py={3}
            maxW="560px"
            w="100%"
          >
            <Code
              color="#FF8888"
              bg="transparent"
              fontSize="12px"
              fontFamily="monospace"
              whiteSpace="pre-wrap"
              wordBreak="break-all"
            >
              {msg}
            </Code>
          </Box>
        )}

        <HStack spacing={3}>
          <Button
            size="sm"
            bg="#FF0032"
            color="#FFFFFF"
            _hover={{ bg: "#CC0028" }}
            borderRadius="8px"
            leftIcon={<Icon as={RepeatIcon} boxSize="13px" />}
            onClick={this.handleReload}
          >
            Перезагрузить страницу
          </Button>
          <Button
            size="sm"
            variant="outline"
            borderColor="#2A2E36"
            color="#888"
            _hover={{ borderColor: "#555", color: "#FFFFFF" }}
            borderRadius="8px"
            onClick={this.handleReset}
          >
            На главную
          </Button>
        </HStack>
      </Flex>
    );
  }
}

export default ErrorBoundary;
