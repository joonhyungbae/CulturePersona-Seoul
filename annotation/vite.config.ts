import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { annoPlugin } from "./server/annoPlugin";

// Deploy (GitHub Pages) is static files + Apps Script. The server block here is for local testing.
//   VITE_BASE      Pages path (the workflow sets /sfac_persona_anno/). Local is /.
//   VITE_ENDPOINT  Apps Script web app URL (the workflow supplies a repository variable). Empty means local /api.
//   ANNO_HOST      Bind the local server to a tailscale IP only. Default 127.0.0.1.
//   ANNO_PORT      Default 5180 (dev mode dev_ux.sh uses 5183).
//   ANNO_HTTPS_CERT / ANNO_HTTPS_KEY  Serve https — over http from outside, the browser blocks unsealing (Web Crypto).
const host = process.env.ANNO_HOST ?? "127.0.0.1";
const cert = process.env.ANNO_HTTPS_CERT;
const key = process.env.ANNO_HTTPS_KEY;

export default defineConfig({
  base: process.env.VITE_BASE ?? "/",
  plugins: [react(), annoPlugin()],
  server: {
    host,
    port: Number(process.env.ANNO_PORT ?? 5180),
    https: cert && key ? { cert: readFileSync(cert), key: readFileSync(key) } : undefined,
    strictPort: true,
    watch: { usePolling: true, interval: 1000, ignored: ["**/node_modules/**", "**/dist/**"] },
    fs: { strict: true, allow: [fileURLToPath(new URL(".", import.meta.url))], deny: [".env", ".env.*"] },
  },
});
