/**
 * XAI-Enhanced Entry Point
 *
 * Drop-in replacement for main.jsx that loads the XAI-enhanced App.
 * The original main.jsx remains completely untouched.
 *
 * To activate: in index.html, change the script src from main.jsx to xai_main.jsx
 * Or rename this file to main.jsx and rename the original main.jsx to main_original.jsx.
 */

import React from "react";
import ReactDOM from "react-dom/client";
import "./styles.css";
import "./xai.css";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import XaiApp from "./XaiApp.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary>
      <XaiApp />
    </ErrorBoundary>
  </React.StrictMode>
);
