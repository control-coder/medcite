import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests', outputDir: '../.cache/implementation/browser',
  use: { baseURL: process.env.MEDIDIAG_WEB_URL || 'http://127.0.0.1:8400',
    headless: true, channel: 'msedge' },
});
