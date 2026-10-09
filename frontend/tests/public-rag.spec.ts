import { mkdir } from 'node:fs/promises';
import { test, expect } from '@playwright/test';

test.skip(process.env.MEDIDIAG_TEST_PROVIDER !== 'retrieval_mock', '需显式启动真实检索/模拟生成 worker');

test('公开中文检索、来源、刷新恢复和无证据反馈', async ({ page }) => {
  await mkdir('../artifacts/visual', {recursive: true});
  await page.goto('/app/');
  await page.getByRole('button', {name: '填入公开检索示例'}).click();
  await page.getByRole('checkbox').check();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({path: '../artifacts/visual/public-rag-form-desktop.png', fullPage: true});
  await page.getByRole('button', {name: '提交并检索证据'}).click();
  await expect(page).toHaveURL(/#\/cases\/case_/);
  const url = page.url(); await page.reload(); await expect(page).toHaveURL(url);
  await page.getByRole('link', {name: '查看结果与证据'}).click({timeout: 30000});
  await expect(page.getByText('真实检索 / 模拟生成（无模型与 NLI）', {exact: true})).toBeVisible();
  await expect(page.locator('.evidence').first()).toContainText('室外空气');
  await expect(page.locator('.evidence a').first()).toHaveAttribute('href', /^https:\/\/www.who.int\/zh\//);
  await expect(page.locator('.claim').first()).toContainText('不代表问题已获解答');
  await page.screenshot({path: '../artifacts/visual/public-rag-desktop.png', fullPage: true});
  await page.getByRole('link', {name: '咨询记录', exact: true}).click();
  await expect(page.locator('.history-item')).toHaveCount(1);
  await page.getByRole('link', {name: '开始咨询', exact: true}).click();
  await page.getByLabel('症状与问题').fill('模拟提问：量子纠缠计算芯片');
  await page.getByLabel('持续时间').fill('模拟');
  await page.getByRole('checkbox').check();
  await page.getByRole('button', {name: '提交并检索证据'}).click();
  await page.getByRole('link', {name: '查看结果与证据'}).click({timeout: 30000});
  await expect(page.locator('.evidence')).toHaveCount(0);
  await expect(page.locator('.claim')).toHaveCount(0);
  await expect(page.getByText('没有可展示的证据，不作确定性结论。')).toBeVisible();
  await page.setViewportSize({width: 390, height: 844});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path: '../artifacts/visual/public-rag-empty-mobile.png', fullPage: true});
});
