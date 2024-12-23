import React from "react";
import MenuComponent from "../components/MenuComponent";
import face from "./../images/face.svg";
import { VStack, HStack, Box, Image, Flex } from "@chakra-ui/react";
import useWindowDimensions from "../hooks/window_dimensions";


const MainPage = () => {

    const { width } = useWindowDimensions();

    return (
        <Flex
            flexDirection="column"
            justify="space-between"
            align="center"
            minHeight="100%"
            width={width}
            height="100vh"
            padding={0}
            bg="#0A050D"
        >
            <VStack
                align="center"
                justify="center"
                bg="transparent"
            >
                <HStack
                    w="100%" // Занимает всю ширину
                    justify="space-between" // Один элемент слева, другой справа
                >
                    {/* Левый элемент */}
                    <Box>
                        <MenuComponent
                            isHorizontal={false}
                            showTitle={true} 
                            buttons={[
                                { label: "Documentation", path: "/documentation" },
                                { label: "Query", path: "/query" },
                            ]}
                        />
                    </Box>

                    {/* Правый элемент */}
                    <Box>
                        <Image src={face} boxSize="900px" alt="Face" />
                    </Box>
                </HStack>
            </VStack>
        </Flex>
    );
};

export default MainPage;