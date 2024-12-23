import React from "react";
import { Box, Text } from "@chakra-ui/react";


const DocFieldComponent = ({ children }) => {
    return (
        <Box
            position="relative"
            width="1308px"
            height="989px"
            borderRadius="4px"
            bg="#1D2023"
            overflow="hidden"
        >
            {/* Градиентный слой */}
            <Box
                position="absolute"
                width="4992.08px"
                height="4222.2px"
                left="-977px"
                top="-1999.79px"
                bgGradient="linear(44.75deg, #0085FF 9.37%, rgba(0, 133, 255, 0) 47.32%), linear(174.38deg, rgba(127, 140, 153, 0.35) 38.55%, rgba(127, 140, 153, 0) 92.85%)"
                transform="rotate(-0.04deg)"
                borderRadius="4px"
            />
            {/* Контейнер для содержимого */}
            <Box
                position="absolute"
                width="1306px"
                height="987px"
                top="1px"
                left="1px"
                background="#1D2023"
                borderRadius="4px"
                zIndex="1"
                display="flex" // Флекс для центрирования
                justifyContent="center"
                alignItems="center"
                color="white" // Текст белого цвета
            >
                {children || <Text fontSize="xl">Your content goes here</Text>}
            </Box>
        </Box>
    );
};

export default DocFieldComponent;