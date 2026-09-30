/**
 * XAI-Enhanced Vite Configuration
 *
 * Uses xai_index.html as the entry point to load the XAI-enhanced app.
 * The original vite.config.js remains completely untouched.
 *
 * Usage: vite --config vite.xai.config.js
 */

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// Dev-server plugin: serve xai_index.html for all SPA navigation requests
const xaiIndexPlugin = {
  name: "xai-index-html",
  configureServer(server) {
    server.middlewares.use((req, _res, next) => {
      if (req.url === "/" || req.url === "/index.html") {
        req.url = "/xai_index.html";
      }
      next();
    });
  },
};

export default defineConfig({
  plugins: [react(), xaiIndexPlugin],
  build: {
    rollupOptions: {
      input: path.resolve(__dirname, "xai_index.html"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
    },
  },
});
