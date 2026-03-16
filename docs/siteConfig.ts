import fs from "node:fs";
import path from "node:path";

export type SiteConfig = {
  siteUrl: string;
  siteRootUrl: string;
  canonicalUrl: string;
  canonicalRootUrl: string;
  ogImageUrl: string;
  repoUrl: string;
  indexable: boolean;
  robotsContent: string;
};

function normalizeUrl(value: string): string {
  return new URL(value).toString().replace(/\/$/, "");
}

function withTrailingSlash(value: string): string {
  return `${value}/`;
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value == null || value === "") {
    return fallback;
  }
  const normalized = value.trim().toLowerCase();
  if (normalized === "true" || normalized === "1" || normalized === "yes") {
    return true;
  }
  if (normalized === "false" || normalized === "0" || normalized === "no") {
    return false;
  }
  throw new Error(`Invalid boolean value '${value}'`);
}

function detectRepoDefaults(
  rootDir: string,
): Partial<SiteConfig> & { repoUrl?: string } {
  const repoName = path.basename(path.resolve(rootDir, ".."));

  if (repoName === "ptspy") {
    return {
      siteUrl: "https://ptspy.org",
      canonicalUrl: "https://ptspy.org",
      repoUrl: "https://github.com/williamngan/ptspy",
      indexable: true,
    };
  }

  return {};
}

export function resolveSiteConfig(
  env: Record<string, string>,
  mode: string,
  rootDir: string,
): SiteConfig {
  const isDev = mode === "development";
  const repoDefaults = detectRepoDefaults(rootDir);
  const siteSource =
    env.VITE_SITE_URL ||
    repoDefaults.siteUrl ||
    (isDev ? "http://localhost:5173" : "");
  if (!siteSource) {
    throw new Error(
      "VITE_SITE_URL is required for non-development docs builds when repo defaults are unavailable",
    );
  }

  const siteUrl = normalizeUrl(siteSource);
  const canonicalUrl = normalizeUrl(
    env.VITE_CANONICAL_URL || repoDefaults.canonicalUrl || siteUrl,
  );
  const repoUrl = normalizeUrl(
    env.VITE_REPO_URL ||
      repoDefaults.repoUrl ||
      "https://github.com/williamngan/ptspy",
  );
  const ogImageUrl = normalizeUrl(
    env.VITE_OG_IMAGE_URL || `${siteUrl}/og-ptspy-v0.1.0.png`,
  );
  const repoIndexable = repoDefaults.indexable;
  const defaultIndexable =
    repoIndexable != null ? repoIndexable : !isDev && siteUrl === canonicalUrl;
  const indexable = parseBoolean(env.VITE_INDEXABLE, defaultIndexable);
  const robotsContent =
    env.VITE_ROBOTS_CONTENT ||
    (indexable
      ? "index,follow,max-image-preview:large"
      : "noindex,nofollow,noarchive,max-image-preview:large");

  return {
    siteUrl,
    siteRootUrl: withTrailingSlash(siteUrl),
    canonicalUrl,
    canonicalRootUrl: withTrailingSlash(canonicalUrl),
    ogImageUrl,
    repoUrl,
    indexable,
    robotsContent,
  };
}

function renderRobots(config: SiteConfig): string {
  if (!config.indexable) {
    return "User-agent: *\nDisallow: /\n";
  }

  return [
    "User-agent: *",
    "Allow: /",
    "Allow: /docs.md",
    "Allow: /llms.txt",
    "Allow: /llms-full.txt",
    "",
    `Sitemap: ${config.siteUrl}/sitemap.xml`,
    "",
  ].join("\n");
}

function renderSitemap(config: SiteConfig): string {
  const urls = config.indexable
    ? [
        config.siteRootUrl,
        `${config.siteUrl}/start`,
        `${config.siteUrl}/docs`,
        `${config.siteUrl}/docs.md`,
        `${config.siteUrl}/llms.txt`,
        `${config.siteUrl}/llms-full.txt`,
      ]
    : [];

  const lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...urls.flatMap((url) => [`  <url>`, `    <loc>${url}</loc>`, `  </url>`]),
    "</urlset>",
    "",
  ];
  return lines.join("\n");
}

function renderLlms(config: SiteConfig): string {
  return [
    "# Pts.py",
    "",
    "> A Python library for visual composability in code. Built on numpy and Skia. Python 3.10+.",
    "",
    "Pts.py explores how visual forms can be composed from simple, atomic operations. It combines functional programming patterns (partial application, pipelines) with numpy-vectorized geometry and Skia-backed rendering.",
    "",
    "Design principles:",
    "- Points as primitives: P(), P2(), P3() create numpy arrays representing coordinates",
    "- Functional programming features like `@partial` decorator + `TBD` placeholder enable currying. `pipeline(a, b, c)` or `F(a) >> b >> c` chains functions. Supports `lazy=True` and `trace=True`.",
    "- Vectorized operations: numpy throughout, no Python loops in hot paths",
    "",
    "## Documentation",
    "",
    `- [Full API documentation with guide](${config.siteUrl}/llms-full.txt): Complete reference with quickstart guide and API docs`,
    `- [API reference (markdown)](${config.siteUrl}/docs.md): Auto-generated API documentation`,
    `- [Quick start guide](${config.siteUrl}/start): Interactive getting started tutorial`,
    `- [Interactive documentation](${config.siteUrl}/docs): Searchable API docs with examples`,
    `- [GitHub repository](${config.repoUrl}): Source code`,
    "",
    "## Optional",
    "",
    `- [Interactive notebooks](${config.repoUrl}/tree/main/notebooks): Jupyter notebooks with examples`,
    "- [PyPI package](https://pypi.org/project/ptspy/): Install via pip",
    "",
  ].join("\n");
}

export function syncSitePublicFiles(rootDir: string, config: SiteConfig): void {
  const publicDir = path.join(rootDir, "public");
  fs.writeFileSync(path.join(publicDir, "robots.txt"), renderRobots(config));
  fs.writeFileSync(path.join(publicDir, "sitemap.xml"), renderSitemap(config));
  fs.writeFileSync(path.join(publicDir, "llms.txt"), renderLlms(config));
}

export function transformSiteHtml(html: string, config: SiteConfig): string {
  return html
    .replaceAll("__CANONICAL_URL__", config.canonicalRootUrl)
    .replaceAll("__SITE_URL__", config.siteRootUrl)
    .replaceAll("__OG_IMAGE_URL__", config.ogImageUrl)
    .replaceAll("__ROBOTS_CONTENT__", config.robotsContent);
}
