import { mkdir } from 'node:fs/promises';
import { test, expect } from '@playwright/test';

test.skip(process.env.MEDIDIAG_TEST_PROVIDER !== 'retrieval_mock', '需显式启动真实检索/模拟生成 worker');

test('公开中文检索、来源、刷新恢复和无证据反馈', async ({ page }) => {
  await mkdir('../artifacts/visual', {recursive: true});
  await page.goto('/app/');
  await page.getByRole('button', {name: '通风如何改善室内空气质量？'}).click();
  await page.getByRole('checkbox').check();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({path: '../artifacts/visual/public-rag-form-desktop.png', fullPage: true});
  await page.getByRole('button', {name: '提交问题'}).click();
  await expect(page).toHaveURL(/#\/cases\/case_/);
  const url = page.url(); await page.reload(); await expect(page).toHaveURL(url);
  await expect(page.getByRole('heading', {name: '回答', exact: true})).toBeVisible({timeout: 30000});
  await expect(page.locator('.run-meta')).toContainText('离线演示 · 真实检索，模拟生成');
  await expect(page.locator('.evidence').first()).toContainText('室外空气');
  await expect(page.locator('.evidence a').first()).toHaveAttribute('href', /^https:\/\/www.who.int\/zh\//);
  await expect(page.locator('.claim').first()).toContainText('不代表问题已获解答');
  await page.locator('.cite-ref').first().click();
  await expect(page.locator('.evidence--active')).toHaveCount(1);
  await page.screenshot({path: '../artifacts/visual/public-rag-desktop.png', fullPage: true});
  await page.getByRole('link', {name: '历史记录', exact: true}).click();
  await expect(page.locator('.history-item')).toHaveCount(1);
  await expect(page.locator('.history-q').first()).toHaveText('通风如何改善室内空气质量？');
  await page.getByRole('link', {name: '提问', exact: true}).click();
  await page.getByLabel('你的问题').fill('模拟提问：量子纠缠计算芯片');
  await page.getByRole('checkbox').check();
  await page.getByRole('button', {name: '提交问题'}).click();
  await expect(page.locator('.abstain')).toContainText('证据不足', {timeout: 30000});
  await expect(page.locator('.evidence')).toHaveCount(0);
  await expect(page.locator('.claim')).toHaveCount(0);
  await expect(page.locator('.sources')).toHaveCount(0);
  await page.setViewportSize({width: 390, height: 844});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path: '../artifacts/visual/public-rag-empty-mobile.png', fullPage: true});
});
