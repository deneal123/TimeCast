import { extendTheme } from "@chakra-ui/react";

const theme = extendTheme({
  config: {
    initialColorMode: "dark",
    useSystemColorMode: false,
  },
  fonts: {
    body: `Inter, Montserrat, sans-serif`,
    heading: `Inter, sans-serif`,
    mono: `"JetBrains Mono", "Fira Code", monospace`,
  },
  colors: {
    // Фоны
    surface0: "#0A050D",   // main page bg
    surface1: "#0D1017",   // deepest panel / textarea
    surface2: "#0F1218",   // section header
    surface3: "#141820",   // card / panel bg
    surface4: "#1A1D21",   // header / footer

    // Borders
    border1: "#FFFFFF0F",  // hairline
    border2: "#2A2E36",    // standard border
    border3: "#444444",    // hover border

    // Accents
    brand:  "#FF0032",
    yellow: "#FFBF00",
    cyan:   "#00B5D8",
    green:  "#48BB78",
    purple: "#9F7AEA",

    // Text
    textPrimary:   "#FFFFFF",
    textSecondary: "#AAAAAA",
    textMuted:     "#666666",
    textDim:       "#444444",

    // Legacy aliases (keep for backward compat)
    menu_gray:  "#CCC3C2",
    main_dark:  "#333",
    light_dark: "#666",
    main_yellow: "#FFBF00",
    main_red:   "#FF0032",
    menu_white: "#F8F8F8",
    date_gray:  "#A9A9A9",
    menu_mts:   "#0F1218",
  },
  components: {
    Tooltip: {
      baseStyle: {
        bg: "#1A1D21",
        color: "#FFFFFF",
        border: "1px solid #2A2E36",
        borderRadius: "8px",
        fontSize: "12px",
        px: "10px",
        py: "6px",
      },
    },
    Button: {
      variants: {
        menu_yellow: {
          border: "1px solid",
          borderColor: "yellow",
          borderRadius: "8px",
          background: "transparent",
          color: "yellow",
          _hover: { bg: "#FFBF0018" },
        },
        menu_red: {
          border: "1px solid",
          borderColor: "brand",
          borderRadius: "8px",
          background: "transparent",
          color: "brand",
          _hover: { bg: "#FF003218" },
        },
      },
    },
  },
  styles: {
    global: {
      body: {
        bg: "#0F1218",
        color: "#FFFFFF",
      },
      "::-webkit-scrollbar": {
        width: "6px",
        height: "6px",
      },
      "::-webkit-scrollbar-track": {
        bg: "#0D1017",
      },
      "::-webkit-scrollbar-thumb": {
        bg: "#2A2E36",
        borderRadius: "3px",
      },
      "::-webkit-scrollbar-thumb:hover": {
        bg: "#444",
      },
    },
  },
});

export default theme;
