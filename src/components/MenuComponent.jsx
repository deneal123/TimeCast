// MenuComponent.jsx
import React from 'react';
import { Flex, Button, Text } from "@chakra-ui/react";
import { useNavigate } from "react-router-dom";

function MenuComponent({
    isHorizontal = false,
    buttons = [],
    showTitle = true,
    hideButtons = false,
    buttonWidth = "200px",
    buttonHeight = "50px",
    disableNavigate = false, // Новый пропс, который отключает navigate
    onClickActions = {} // Ожидаем onClickActions как пропс
}) {
    const navigate = useNavigate(); // Используем useNavigate для навигации

    // Функция для обработки нажатий на кнопки
    const handleButtonClick = (label) => {
        if (!disableNavigate) {
            // Переход по маршруту, если disableNavigate не установлен
            navigate(label.path);
        }
        // Выполняем действие для кнопки, если оно задано в onClickActions
        if (onClickActions[label]) {
            onClickActions[label](); // Вызов действия, не связанного с навигацией
        }
    };

    return (
        <Flex
            direction={isHorizontal ? "row" : "column"}
            bg="transparent"
            padding={25}
            spacing="20px"
            flexGrow={1}
            align="center"
            justify="center"
        >
            {/* Display the title "Menu" if showTitle is true */}
            {showTitle && (
                <Text
                    fontFamily="Inter"
                    fontWeight="0"
                    fontSize="42px"
                    lineHeight="42px"
                    color="#FFFFFF"
                    mb={isHorizontal ? 0 : 5}
                >
                    Menu
                </Text>
            )}

            {/* Генерация кнопок, если hideButtons не равно true */}
            {!hideButtons && (
                <>
                    {buttons.map(({ label, path }, index) => (
                        <Button
                            key={index}
                            fontSize="Inter"
                            fontWeight="0"
                            width={buttonWidth}
                            height={buttonHeight}
                            bg="#FF0032"
                            color="#FFFFFF"
                            borderRadius="10px"
                            _hover={{ bg: "#D0021B" }}
                            _active={{ transform: "scale(0.95)" }}
                            onClick={() => handleButtonClick({ label, path })} // Передаем объект с label и path
                            mx={isHorizontal ? 2 : 0}
                            my={isHorizontal ? 0 : 2}
                        >
                            {label}
                        </Button>
                    ))}
                </>
            )}
        </Flex>
    );
}

export default MenuComponent;
