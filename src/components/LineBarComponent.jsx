import React from "react";
import { Box, Button, Icon, HStack } from "@chakra-ui/react";
import { FaHandPaper } from "react-icons/fa"; // ���������� ������ ��� �����

const LineBarComponent = () => {
  return (
    <Box
      position="relative"
      width="80%" // ������� ������ ����������
      maxWidth="1308px" // ������������ ������
      height="96px"
      background="#151618"
      borderRadius="6px"
      padding="10px"
      margin="0 auto"
      boxShadow="md" // �������� ���� ��� �������
    >
      {/* ������ � ���� */}
      <HStack spacing="20px" align="center" justify="center">
        {/* ������ */}
        <Button
          width="44px"
          height="44px"
          bg="#FF0032"
          borderRadius="8px"
          _hover={{ bg: "#D0021B" }}
        >
          <Icon as={FaHandPaper} color="white" boxSize="20px" />
        </Button>
        <Button
          width="44px"
          height="44px"
          bg="rgba(255, 255, 255, 0.7)"
          borderRadius="8px"
          _hover={{ bg: "rgba(255, 255, 255, 1)" }}
        >
          Scroll
        </Button>
        <Box
          position="relative"
          width="44px"
          height="44px"
          background="#FFFFFF"
          borderRadius="8px"
          display="flex"
          justifyContent="center"
          alignItems="center"
        >
          <Box
            width="18.33px"
            height="21.77px"
            background="#FF0032"
            borderRadius="4px"
            position="absolute"
          />
        </Box>
      </HStack>
    </Box>
  );
};

export default LineBarComponent;
