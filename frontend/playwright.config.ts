import { defineConfig } from '@playwright/test';
// 本机默认复用已安装的 Edge；CI 设置 MEDIDIAG_BROWSER_CHANNEL=chromium 使用 Playwright 自带 Chromium。
const channel = process.env.MEDIDIAG_BROWSER_CHANNEL || 'msedge';
export default defineConfig({
  testDir: './tests', outputDir: '../.cache/implementation/browser',
  use: { baseURL: process.env.MEDIDIAG_WEB_URL || 'http://127.0.0.1:8400',
    headless: true, channel },
});
