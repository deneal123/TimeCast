export const PLOT_LAYOUT_BASE = {
  autosize: true,
  paper_bgcolor: "#141820",
  plot_bgcolor: "#0D1017",
  font: { color: "#888", size: 12, family: "Inter, sans-serif" },
  margin: { l: 60, r: 20, t: 16, b: 50 },
  xaxis: {
    gridcolor: "#1E2330",
    linecolor: "#2A2E36",
    zerolinecolor: "#2A2E36",
    tickfont: { color: "#666" },
  },
  yaxis: {
    title: { text: "Значение", font: { color: "#555", size: 12 } },
    gridcolor: "#1E2330",
    linecolor: "#2A2E36",
    zerolinecolor: "#2A2E36",
    tickfont: { color: "#666" },
  },
  legend: {
    orientation: "h",
    y: -0.15,
    bgcolor: "transparent",
    borderwidth: 0,
    font: { color: "#888", size: 12 },
  },
  hovermode: "x unified",
  hoverlabel: { bgcolor: "#1A1D21", bordercolor: "#2A2E36", font: { color: "#FFFFFF" } },
};

export const PLOT_CONFIG = {
  responsive: true,
  displayModeBar: "hover",
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "resetScale2d"],
  toImageButtonOptions: { format: "png", filename: "timecast_chart", scale: 2 },
};
