import React from "react";
import ReactDOM from "react-dom/client";
import "./pacs_glass.css";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import PacsApp from "./PacsApp.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary>
      <PacsApp />
    </ErrorBoundary>
  </React.StrictMode>
);
