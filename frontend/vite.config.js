import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Read configuration from environment (injected by deploy.sh)
const BACKEND_HOST = process.env.VITE_BACKEND_HOST || "127.0.0.1";
const BACKEND_PORT = parseInt(process.env.VITE_BACKEND_PORT || "9000");
const NUM_WORKERS = parseInt(process.env.VITE_NUM_WORKERS || "1");

console.log(`🚀 Proxying API to http://${BACKEND_HOST}:${BACKEND_PORT} (${NUM_WORKERS} workers)`);

export default defineConfig({
  plugins: [react()],
  preview: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: [".loca.lt", ".trycloudflare.com"],
    proxy: {
      "/api": {
        target: `http://${BACKEND_HOST}:${BACKEND_PORT}`,
        changeOrigin: true,
        router: (req) => {
          if (NUM_WORKERS <= 1) return `http://${BACKEND_HOST}:${BACKEND_PORT}`;
          const offset = Math.floor(Math.random() * NUM_WORKERS);
          const port = BACKEND_PORT + offset;
          return `http://${BACKEND_HOST}:${port}`;
        },
      },
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: [".loca.lt", ".trycloudflare.com"],
    proxy: {
      "/api": {
        target: `http://${BACKEND_HOST}:${BACKEND_PORT}`,
        changeOrigin: true,
        router: (req) => {
          if (NUM_WORKERS <= 1) return `http://${BACKEND_HOST}:${BACKEND_PORT}`;
          const offset = Math.floor(Math.random() * NUM_WORKERS);
          const port = BACKEND_PORT + offset;
          return `http://${BACKEND_HOST}:${port}`;
        },
      },
    },
  },
});
