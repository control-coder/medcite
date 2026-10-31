import { existsSync } from 'node:fs';
import { mkdir, readFile } from 'node:fs/promises';
import { test, expect, type Page } from '@playwright/test';

// 生成 README 截图：需要已启动真实模型（mimo_grounded + 生成式回答配置）的 API 与 worker，
// 显式设置 MEDIDIAG_README_SHOTS=1 才运行。各用例共用同一份匿名会话（storageState），
// 便于中途换 worker 的预算账本后继续，历史记录页能看到前面提交的问题。
test.skip(process.env.MEDIDIAG_README_SHOTS !== '1', '只在显式生成 README 截图时运行，会触发真实模型调用');
test.describe.configure({mode: 'serial'});
// 首次启动 worker 要加载语料索引与重排模型，CPU 上一题可能要几分钟
test.setTimeout(1200000);

const out = '../artifacts/visual/readme-';
const state = '../.cache/implementation/readme-shots/state.json';
test.use({viewport: {width: 1280, height: 800}});

test.beforeEach(async ({ page }) => {
  if (existsSync(state)) await page.context().addCookies(JSON.parse(await readFile(state, 'utf-8')).cookies);
});

async function ask(page: Page, question: string) {
  await page.goto('/app/');
  await page.getByLabel('你的问题').fill(question);
  await page.getByRole('checkbox').check();
  await page.getByRole('button', {name: '提交问题'}).click();
  await expect(page).toHaveURL(/#\/cases\/case_/);
}

test.afterEach(async ({ page }) => {
  await mkdir('../.cache/implementation/readme-shots', {recursive: true});
  await page.context().storageState({path: state});
});

test('科普问题：提问、进度与生成式回答', async ({ page }) => {
  await mkdir('../artifacts/visual', {recursive: true});
  await page.goto('/app/');
  await page.getByRole('button', {name: '登革热有哪些症状，怎样预防？'}).click();
  await page.getByRole('checkbox').check();
  await page.screenshot({path: out + 'ask.png'});
  await page.getByRole('button', {name: '提交问题'}).click();
  await expect(page.locator('.steps li.current')).toBeVisible();
  // 等到检索之后的步骤再截进度图
  await expect(page.locator('.steps li.done').nth(1)).toBeVisible({timeout: 600000});
  await page.screenshot({path: out + 'progress.png'});
  await expect(page.getByRole('heading', {name: '回答', exact: true})).toBeVisible({timeout: 600000});
  await expect(page.locator('.claim').first()).toBeVisible();
  await page.locator('.cite-ref').first().click();
  await page.waitForTimeout(800);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({path: out + 'answer.png', fullPage: true});
});

test('急症问题：路由直接给出 120 提示', async ({ page }) => {
  await ask(page, process.env.MEDIDIAG_README_EMERGENCY || '孩子现在抽搐不止，叫不醒，怎么办？');
  await expect(page.locator('.abstain--urgent')).toBeVisible({timeout: 120000});
  await page.screenshot({path: out + 'emergency.png', fullPage: true});
});

test('个体诊疗问题：拒绝给出个体意见', async ({ page }) => {
  await ask(page, process.env.MEDIDIAG_README_INDIVIDUAL || '我妈血压高，降压药吃了一年了，现在能不能停药？');
  await expect(page.locator('.abstain')).toBeVisible({timeout: 120000});
  await page.screenshot({path: out + 'individual.png', fullPage: true});
});

test('历史记录', async ({ page }) => {
  await page.goto('/app/#/history');
  await expect(page.locator('.history-item').first()).toBeVisible();
  await page.screenshot({path: out + 'history.png'});
});
