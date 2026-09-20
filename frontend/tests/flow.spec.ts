import { mkdir } from 'node:fs/promises';
import { test, expect } from '@playwright/test';

test('离线咨询闭环、刷新恢复与移动布局', async ({ page }) => {
  await mkdir('../artifacts/visual', {recursive: true});
  await page.goto('/app/');
  await expect(page.getByRole('heading', {name: '新建辅助分析'})).toBeVisible();
  await page.getByRole('button', {name: '填入模拟示例'}).click();
  await page.getByRole('checkbox').check();
  await page.getByRole('button', {name: '提交并检索证据'}).click();
  await expect(page).toHaveURL(/#\/cases\/case_/);
  const url = page.url(); await page.reload(); await expect(page).toHaveURL(url);
  await page.getByRole('link', {name: '查看结果与证据'}).click({timeout: 30000});
  await expect(page.getByRole('heading', {name: '辅助分析', exact: true})).toBeVisible();
  await expect(page.locator('.evidence')).toHaveCount(1);
  await expect(page.getByRole('heading', {name: '运行观测'})).toBeVisible();
  await expect(page.getByText(/费用：未计价/)).toBeVisible();
  await page.screenshot({path: '../artifacts/visual/react-result-desktop.png', fullPage: true});
  await page.getByRole('link', {name: '咨询记录', exact: true}).click();
  await expect(page.locator('.history-item')).toHaveCount(1);
  await page.setViewportSize({width: 390, height: 844}); await page.goto('/app/');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path: '../artifacts/visual/react-home-mobile.png', fullPage: true});
  await expect(page.getByRole('heading', {name: '新建辅助分析'})).toBeVisible();
});
