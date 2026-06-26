import { TimeIcon, CheckCircleIcon, WarningIcon } from "@chakra-ui/icons";

/**
 * Single source of truth for task status display metadata.
 * Shared by tasks_page.jsx and TaskStatusPanel.jsx.
 *
 * Fields:
 *   color   — hex for text/spinner
 *   scheme  — Chakra colorScheme for Badge
 *   icon    — Chakra icon component (or null for "running")
 *   label   — Russian display label
 *   dot     — hex for the small status indicator dot in the task list
 */
export const STATUS_META = {
  pending: {
    color:  "#888",
    scheme: "gray",
    icon:   TimeIcon,
    label:  "В очереди",
    dot:    "#888",
  },
  running: {
    color:  "#FFBF00",
    scheme: "yellow",
    icon:   null,
    label:  "Выполняется",
    dot:    "#FFBF00",
  },
  done: {
    color:  "#48BB78",
    scheme: "green",
    icon:   CheckCircleIcon,
    label:  "Готово",
    dot:    "#48BB78",
  },
  failed: {
    color:  "#FF8888",
    scheme: "red",
    icon:   WarningIcon,
    label:  "Ошибка",
    dot:    "#FF8888",
  },
};
