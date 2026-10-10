import { mkdir, writeFile } from 'node:fs/promises';
import { test, expect } from '@playwright/test';

test.skip(process.env.MEDIDIAG_ALLOW_LIVE !== 'mimo-v2.6-flash', '只有显式授权脚本可执行真实模型验收');
test.setTimeout(240000);

test('真实 MiMo 页面闭环、保守弃答与空检索', async ({ page }) => {
  const records: unknown[] = [];
  const cases = [
    {id: 'ventilation', question: '公开资料如何定义通风？', outcome: 'ready'},
    {id: 'heat', question: '高温时可以把孩子留在停放的车辆中吗？', outcome: 'ready'},
    {id: 'unsupported', question: '疫苗抗体滴度的具体阈值', outcome: 'insufficient_evidence'},
    {id: 'empty', question: '模拟提问：量子纠缠计算芯片', outcome: 'insufficient_evidence'},
  ];
  await mkdir('../artifacts/visual', {recursive: true});
  for (const item of cases) {
    await page.goto('/app/');
    await page.getByLabel('症状与问题').fill(item.question);
    await page.getByLabel('持续时间').fill('公开科普模拟，不适用');
    await page.getByRole('checkbox').check();
    const workflowResponse = page.waitForResponse(r => r.url().endsWith('/workflow') && r.request().method() === 'POST');
    const started = Date.now();
    await page.getByRole('button', {name: '提交并检索证据'}).click();
    await expect(page).toHaveURL(/#\/cases\/case_/);
    const caseId = page.url().match(/case_[a-z0-9]+/)![0];
    const prefix = '/api/v1/cases/' + caseId;
    const taskId = (await (await workflowResponse).json()).task_id;
    const url = page.url(); await page.reload(); await expect(page).toHaveURL(url);
    let analysis: any;
    await expect.poll(async () => {
      analysis = await (await page.request.get(prefix + '/analysis')).json();
      return analysis.outcome;
    }, {timeout: 65000, intervals: [250, 500]}).not.toBe('processing');
    const elapsedMs = Date.now() - started;
    records.push({id: item.id, question: item.question, expected_outcome: item.outcome,
      elapsed_ms: elapsedMs, case_id: caseId, task_id: taskId, analysis});
    // 即使断言失败，也保留已发生的真实结果；绝不为得到成功而覆盖失败记录。
    await writeFile(process.env.MEDIDIAG_LIVE_CASES!, JSON.stringify(records, null, 2));
    expect.soft(analysis.outcome).toBe(item.outcome);
    expect(analysis.execution_mode).toBe('mimo_grounded');
    await page.getByRole('link', {name: '查看结果与证据'}).click({timeout: 10000});
    await expect(page.getByText('真实检索 / MiMo 受约束摘录（非 NLI 审核）', {exact: true})).toBeVisible();
    if (analysis.outcome === 'ready') {
      await expect(page.locator('.claim').first()).toBeVisible();
      await expect(page.locator('.evidence a').first()).toHaveAttribute('href', /^https:\/\/www.who.int\/zh\//);
      expect(analysis.observation.recorded_input_tokens).toBeGreaterThan(0);
      for (const claim of analysis.claims) {
        expect(analysis.evidence.some((e: any) => e.chunk_id === claim.evidence_ids[0] && e.text === claim.text)).toBe(true);
      }
    } else {
      await expect(page.locator('.claim')).toHaveCount(0);
      expect(analysis.claims).toHaveLength(0);
    }
    if (item.id === 'empty') {
      expect(analysis.evidence).toHaveLength(0);
      expect(analysis.observation.recorded_input_tokens).toBeNull();
      await page.setViewportSize({width: 390, height: 844});
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    }
    await page.screenshot({path: '../artifacts/visual/mimo-live-' + item.id + '.png', fullPage: true});
  }
  await page.getByRole('link', {name: '咨询记录', exact: true}).click();
  await expect(page.locator('.history-item')).toHaveCount(4);
});
