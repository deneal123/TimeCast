import { Flex, HStack, Text, Icon, Link } from "@chakra-ui/react";
import { ExternalLinkIcon } from "@chakra-ui/icons";

const Footer = () => (
  <Flex
    as="footer"
    w="100%"
    h="56px"
    bg="#1A1D21"
    borderTop="1px solid #FFFFFF0F"
    align="center"
    justify="space-between"
    px={[4, 6, 10]}
    flexShrink={0}
  >
    <Text color="#444" fontSize="13px">
      TimeCast © 2024 — Вольхин Данил, ВШЭ
    </Text>
    <HStack spacing={4}>
      <Text color="#333" fontSize="12px">МТС — временные ряды</Text>
      <Link
        href="https://github.com/deneal123/TimeCast"
        isExternal
        color="#555"
        fontSize="12px"
        _hover={{ color: "#FFFFFF" }}
        transition="color 0.2s"
      >
        GitHub <Icon as={ExternalLinkIcon} boxSize="11px" ml={1} />
      </Link>
    </HStack>
  </Flex>
);

export default Footer;
