import { mkdir } from 'node:fs/promises';
import { test, expect } from '@playwright/test';

test.skip(process.env.MEDIDIAG_TEST_WORKER_PAUSED !== '1', '需复验脚本先暂停独立 worker，确定性验证取消反馈');

test('重复点击、待领取取消和 Cookie 丢失反馈', async ({ page, context }) => {
  await mkdir('../artifacts/visual', {recursive: true});
  await page.goto('/app/');
  await page.getByRole('button', {name: '填入公开检索示例'}).click();
  await page.getByRole('checkbox').check();
  let release!: () => void;
  const pending = new Promise<void>(resolve => {release = resolve;});
  let creates = 0;
  // 只延迟浏览器发包，不伪造响应；请求释放后仍由真实 API 写入新数据库。
  await page.route('**/api/v1/consultations', async route => {
    creates += 1;
    await pending;
    await route.continue();
  });
  try {
    await page.getByRole('button', {name: '提交并检索证据'}).dblclick();
    await expect(page.getByRole('button', {name: '正在提交…'})).toBeDisabled();
    await expect.poll(() => creates).toBe(1);
  } finally { release(); }
  await expect(page).toHaveURL(/#\/cases\/case_/);
  expect(creates).toBe(1);
  const taskUrl = page.url();
  const caseId = taskUrl.split('/').pop()!;
  await page.getByRole('button', {name: '取消本次任务'}).click();
  await expect(page.getByRole('heading', {name: '已取消', exact: true})).toBeVisible();
  await page.reload();
  await expect(page.getByRole('heading', {name: '已取消', exact: true})).toBeVisible();
  await expect(page.getByRole('button', {name: '取消本次任务'})).toHaveCount(0);
  await expect(page.getByText('任务已结束，可从咨询记录返回；刷新不会重新执行。')).toBeVisible();
  await expect(page.getByText(/任务由后端持续执行/)).toHaveCount(0);
  const analysis = await context.request.get('/api/v1/cases/' + caseId + '/analysis');
  expect(analysis.status()).toBe(200);
  expect(await analysis.json()).toMatchObject({outcome: 'cancelled', claims: [], evidence: []});
  await page.screenshot({path: '../artifacts/visual/task-cancelled-desktop.png', fullPage: true});
  await page.getByRole('link', {name: '咨询记录', exact: true}).click();
  await expect(page.locator('.history-item')).toHaveCount(1);
  // 清 Cookie 后实际建立新 owner；不恢复旧凭据、不认领旧记录。
  await context.clearCookies();
  await page.goto(taskUrl);
  await page.reload();
  await expect(page.getByRole('alert')).toContainText('任务不存在或不属于当前匿名会话');
  await expect(page.getByText('正在读取任务…', {exact: true})).toHaveCount(0);
  await expect(page.getByRole('alert')).toContainText('清除 Cookie 或换浏览器后无法找回');
  expect((await context.request.get('/api/v1/cases/' + caseId + '/analysis')).status()).toBe(404);
  await page.setViewportSize({width: 390, height: 844});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path: '../artifacts/visual/session-lost-mobile.png', fullPage: true});
  await page.getByRole('link', {name: '咨询记录', exact: true}).click();
  await expect(page.getByText('暂无咨询记录。', {exact: false})).toBeVisible();
  await expect(page.locator('.history-item')).toHaveCount(0);
});
