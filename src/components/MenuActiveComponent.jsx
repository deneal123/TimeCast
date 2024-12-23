import React from 'react';
import { Flex, Button, Text } from "@chakra-ui/react";


function MenuActiveComponent({
    isHorizontal = false,
    buttons = [],
    showTitle = true,
    hideButtons = false,
    buttonWidth = "200px",
    buttonHeight = "50px",
    onClickActions = {} // Expecting onClickActions as a prop
}) {
    // Function for handling button clicks
    const handleButtonClick = (action) => {
        if (onClickActions[action]) {
            onClickActions[action](); // Call the relevant function from onClickActions
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

            {/* Generate buttons if hideButtons is false */}
            {!hideButtons && (
                <>
                    {buttons.map(({ label }, index) => (
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
                            onClick={() => handleButtonClick(label)} // Call handleButtonClick with the label of the button
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

export default MenuActiveComponent;
