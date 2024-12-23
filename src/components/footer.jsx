import { HStack, Text, Flex } from "@chakra-ui/react";

const Footer = () => {
    return (
        <Flex
            backgroundColor="#2C3135"
            width="100%"
            minHeight={["60px", "70px", "80px", "90px", "100px"]}
            padding={["10px", "12px", "15px", "20px", "25px"]}
            justifyContent="center" // Центрирует контент внутри Flex
        >
            <HStack justify="space-between" width="50%">
                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="18px"
                    lineHeight="22px"
                    fontWeight="0"
                    width="308px"
                    textAlign="right"
                >
                    Вольхин Данил ВШЭ
                </Text>
                <HStack
                    onClick={() => {
                        window.scrollTo(0, 0);
                    }}
                >
                    <Text
                        color="#FFFFFF"
                        fontFamily="Inter"
                        fontSize="18px"
                        lineHeight="22px"
                        fontWeight="0"
                        width="400px"
                        textAlign="left"
                    >
                        Домашняя работа по временным рядам | МТС
                    </Text>
                </HStack>
            </HStack>
        </Flex>
    );
};

export default Footer;
