import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  server: {
    host: "127.0.0.1",
    port: 5173,
    open: false,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    target: "es2022",
  },
});
