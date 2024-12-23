import React from "react";
import logo from "./../../images/logo.svg";
import {
    Flex,
    Image,
    Text,
} from "@chakra-ui/react";
import MenuComponent from "../../components/MenuComponent";
import useWindowDimensions from "../../hooks/window_dimensions";



const Header = ({ showMenu = false, hideButtons = false, menuButtons = [] }) => {
    const { width } = useWindowDimensions();

    return (
        <Flex
            as="header"
            position="absolute"
            width={width}
            height="100px"
            left="50%"
            transform="translateX(-50%)"
            top="0"
            bg="#2C3135"
            align="center"
            justify="space-between"
        >
            {/* Логотип и название TimeCast */}
            <Flex
                position="absolute"
                left="76px"
                top="50%"
                transform="translateY(-50%)"
                align="center"
                gap="8px"
            >
                <Image src={logo} boxSize="44px" alt="Logo" />
                <Text
                    fontFamily="Inter"
                    fontWeight="0"
                    fontSize="18px"
                    lineHeight="22px"
                    color="#FFFFFF"
                >
                    TimeCast
                </Text>
            </Flex>

            {/* Условное отображение MenuComponent */}
            {showMenu && (
                <Flex
                    position="absolute"
                    right="76px"
                    top="50%"
                    transform="translateY(-50%)"
                >
                    <MenuComponent
                        isHorizontal={true}
                        showTitle={false}
                        buttonWidth={"150px"}
                        buttonHeight={"44px"}
                        hideButtons={hideButtons}
                        buttons={menuButtons}
                    />
                </Flex>
            )}
        </Flex>
    );
};

export default Header;