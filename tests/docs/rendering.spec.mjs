import fs from "node:fs";
import path from "node:path";

import { expect, test } from "@playwright/test";

const siteRoot = path.resolve("site");

function renderedPages() {
  const pages = [];

  function visit(directory) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const absolutePath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        visit(absolutePath);
        continue;
      }
      if (entry.name !== "index.html") {
        continue;
      }

      const html = fs.readFileSync(absolutePath, "utf8");
      if (html.includes('http-equiv="refresh"')) {
        continue;
      }

      const relativePath = path.relative(siteRoot, absolutePath);
      const route = `/${relativePath.replaceAll(path.sep, "/").replace(/index\.html$/, "")}`;
      pages.push({
        route,
        hasMermaid: html.includes('class="mermaid-source"'),
      });
    }
  }

  visit(siteRoot);
  return pages.sort((left, right) => left.route.localeCompare(right.route));
}

const pages = renderedPages();

test("the strict build contains rendered content pages", () => {
  expect(pages.length).toBeGreaterThan(0);
});

for (const { route, hasMermaid } of pages) {
  test(`${route} renders its primary content`, async ({ page }, testInfo) => {
    await page.route("https://api.github.com/**", (request) => request.abort());
    const response = await page.goto(route, { waitUntil: "domcontentloaded" });

    expect(response?.ok()).toBe(true);
    await expect(page.locator("main h1")).toBeVisible();

    if (hasMermaid) {
      const diagram = page.locator(".mermaid svg");
      await expect(diagram).toBeVisible();
      expect(await diagram.locator(".nodeLabel").count()).toBeGreaterThan(0);
      expect(await diagram.evaluate((svg) => svg.getBoundingClientRect().width)).toBeGreaterThan(0);
    }

    if (testInfo.project.name === "mobile-chromium") {
      const viewportWidth = page.viewportSize()?.width ?? 0;
      const pageWidth = await page.evaluate(() => document.documentElement.scrollWidth);
      expect(pageWidth).toBeLessThanOrEqual(viewportWidth);
    }
  });
}