/**
 * esbuild build for the BibSync Chrome extension.
 *
 * Three independent bundles — Chrome loads each in a different context:
 *   serviceWorker.js  — MV3 background service worker (module type)
 *   contentScript.js  — injected into Overleaf pages (classic IIFE; content
 *                        scripts can't be ES modules)
 *   sidePanel/index.js — the side-panel UI
 *
 * Static assets (manifest.json, the side-panel HTML/CSS, icons) are copied
 * verbatim into dist/.
 */
import * as esbuild from "esbuild";
import { cpSync, mkdirSync, existsSync } from "node:fs";

const watch = process.argv.includes("--watch");
const outdir = "dist";

mkdirSync(outdir, { recursive: true });
mkdirSync(`${outdir}/sidePanel`, { recursive: true });

// Static assets copied as-is.
const copyStatic = () => {
  cpSync("manifest.json", `${outdir}/manifest.json`);
  cpSync("src/sidePanel/index.html", `${outdir}/sidePanel/index.html`);
  cpSync("src/sidePanel/styles.css", `${outdir}/sidePanel/styles.css`);
  if (existsSync("icons")) {
    cpSync("icons", `${outdir}/icons`, { recursive: true });
  }
};

/** @type {esbuild.BuildOptions} */
const common = {
  bundle: true,
  target: "es2022",
  logLevel: "info",
  sourcemap: watch ? "inline" : false,
  minify: !watch,
};

const builds = [
  { entryPoints: ["src/serviceWorker.ts"], outfile: `${outdir}/serviceWorker.js`, format: "esm" },
  { entryPoints: ["src/contentScript.ts"], outfile: `${outdir}/contentScript.js`, format: "iife" },
  { entryPoints: ["src/sidePanel/index.ts"], outfile: `${outdir}/sidePanel/index.js`, format: "iife" },
];

if (watch) {
  copyStatic();
  for (const b of builds) {
    const ctx = await esbuild.context({ ...common, ...b });
    await ctx.watch();
  }
  console.log("[esbuild] watching…");
} else {
  for (const b of builds) {
    await esbuild.build({ ...common, ...b });
  }
  copyStatic();
  console.log("[esbuild] build complete → dist/");
}
