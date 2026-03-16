import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { resolveSiteConfig, syncSitePublicFiles, transformSiteHtml } from "./siteConfig";

// https://vitejs.dev/config/
const rootDir = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, rootDir, "");
  const siteConfig = resolveSiteConfig(env, mode, rootDir);

  if (mode !== "development") {
    syncSitePublicFiles(path.resolve(rootDir), siteConfig);
  }

  return {
    plugins: [
      react(),
      {
        name: "ptspy-site-config",
        transformIndexHtml(html) {
          return transformSiteHtml(html, siteConfig);
        },
      },
    ],
    base: "/",
  };
});
