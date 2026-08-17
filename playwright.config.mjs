import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "tests/docs",
  outputDir: "test-results",
  reporter: [["list"], ["html", { open: "never" }]],
  screenshot: "only-on-failure",
  trace: "retain-on-failure",
  use: {
    baseURL: "http://127.0.0.1:4173",
  },
  webServer: {
    command: "python -m http.server 4173 --bind 127.0.0.1 --directory site",
    port: 4173,
    reuseExistingServer: false,
  },
  projects: [
    {
      name: "desktop-chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "mobile-chromium",
      use: { ...devices["Pixel 5"] },
    },
  ],
});