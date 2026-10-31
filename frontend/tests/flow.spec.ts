import { mkdir } from 'node:fs/promises';
import { test, expect } from '@playwright/test';

test('离线咨询闭环、刷新恢复与移动布局', async ({ page }) => {
  await mkdir('../artifacts/visual', {recursive: true});
  await page.goto('/app/');
  await expect(page.getByRole('heading', {name: '健康科普问答'})).toBeVisible();
  await page.getByRole('button', {name: '通风如何改善室内空气质量？'}).click();
  await page.getByRole('checkbox').check();
  await page.getByRole('button', {name: '提交问题'}).click();
  await expect(page).toHaveURL(/#\/cases\/case_/);
  const url = page.url(); await page.reload(); await expect(page).toHaveURL(url);
  await expect(page.getByRole('heading', {name: '回答', exact: true})).toBeVisible({timeout: 30000});
  await expect(page.locator('.evidence')).toHaveCount(1);
  await expect(page.locator('.run-meta')).toContainText('处理耗时');
  await page.screenshot({path: '../artifacts/visual/react-result-desktop.png', fullPage: true});
  await page.getByRole('link', {name: '历史记录', exact: true}).click();
  await expect(page.locator('.history-item')).toHaveCount(1);
  await page.setViewportSize({width: 390, height: 844}); await page.goto('/app/');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path: '../artifacts/visual/react-home-mobile.png', fullPage: true});
  await expect(page.getByRole('heading', {name: '健康科普问答'})).toBeVisible();
});
