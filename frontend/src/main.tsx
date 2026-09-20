import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

type Case = { case_id: string; status: string; active_task: { status: string } | null };
type Analysis = { case_id: string; status: string; outcome: string; message: string; summary: string | null;
  claims: { claim_id: string; text: string; evidence_ids: string[] }[];
  evidence: { chunk_id: string; text: string; source: string; source_id: string; source_url: string | null; evidence_level: string }[];
  limitations: string[]; risk_warnings: string[]; next_steps: string[]; disclaimer: string; failure_code: string | null };
const states: Record<string, string> = { CREATED: '已提交', NORMALIZED: '检索中', EVIDENCE_RETRIEVED: '证据就绪', PLAN_GENERATED: '分析中', SPECIALIST_REVIEWING: '分析中', ARBITRATION_REVIEWING: '审核中', APPROVED: '整理结果', REPORT_GENERATED: '整理结果', CLOSED_SUCCESS: '已完成', ESCALATED: '处理受阻', REVISION_REQUIRED: '修订中', CLOSED_FAILED: '失败', CLOSED_CANCELLED: '已取消', CLOSED_ESCALATED: '已终止' };
async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch('/api/v1' + path, { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '输入不符合要求，请检查必填项和公开来源。');
  return data as T;
}
const go = (path: string) => { location.hash = path; };
function App() {
  const [route, setRoute] = useState(location.hash.slice(1) || '/');
  useEffect(() => { const change = () => setRoute(location.hash.slice(1) || '/'); window.addEventListener('hashchange', change); return () => window.removeEventListener('hashchange', change); }, []);
  const parts = route.split('/');
  return <><header><a className="brand" href="#/">M<span>+</span> <strong>MediDiag</strong></a><nav aria-label="主导航"><a href="#/" aria-current={route === '/' ? 'page' : undefined}>开始咨询</a><a href="#/history" aria-current={route === '/history' ? 'page' : undefined}>咨询记录</a><a href="/demo">工程工作台 ↗</a></nav><span className="mode">离线 / 工程原型</span></header>
    <main>{parts[1] === 'history' ? <History/> : parts[1] === 'cases' && parts[2] ? <Task key={parts[2] + parts[3]} id={parts[2]} result={parts[3] === 'result'}/> : <Consult/>}</main>
    <footer>公开证据 · 可追踪引用 · 明确局限 <span>仅用于工程演示，不提供真实患者诊疗服务。</span></footer></>;
}
function Consult() {
  const [symptoms, setSymptoms] = useState(''); const [duration, setDuration] = useState('');
  const [background, setBackground] = useState(''); const [kind, setKind] = useState('deidentified_simulation');
  const [source, setSource] = useState(''); const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const submit = async (event: React.FormEvent) => {
    event.preventDefault(); setBusy(true); setError('');
    const payload = {symptoms, duration, background, input_kind: kind, source_ref: source || null, non_sensitive_confirmed: confirmed};
    // 只保存幂等键；不在浏览器持久保存症状与背景。
    const key = sessionStorage.getItem('medidiag-submit-key') || crypto.randomUUID(); sessionStorage.setItem('medidiag-submit-key', key);
    try { const item = await api<Case>('/consultations', {method: 'POST', headers: {'Idempotency-Key': key}, body: JSON.stringify(payload)}); sessionStorage.removeItem('medidiag-submit-key'); go('/cases/' + item.case_id); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  return <><section className="intro"><p className="eyebrow">EVIDENCE FIRST / 证据优先</p><h1>让每一条分析，<br/>都有据可循。</h1><p>从公开资料与模拟问题开始，了解证据、来源和不确定性。</p></section><div className="columns"><section className="card"><div className="section-title"><h2>新建辅助分析</h2><span>01 / 提交问题</span></div><form onSubmit={submit}>
    <label>输入来源<select value={kind} onChange={e => setKind(e.target.value)}><option value="deidentified_simulation">脱敏模拟输入</option><option value="public_dataset">公开数据集</option></select></label>
    {kind === 'public_dataset' && <label>公开来源标识<input required maxLength={128} value={source} onChange={e => setSource(e.target.value)} placeholder="数据集名称与样本标识"/></label>}
    <label>症状与问题<textarea required minLength={10} maxLength={8000} value={symptoms} onChange={e => setSymptoms(e.target.value)} placeholder="描述模拟症状与想了解的问题，不要包含真实身份信息。" rows={5}/></label>
    <div className="form-row"><label>持续时间<input required maxLength={200} value={duration} onChange={e => setDuration(e.target.value)} placeholder="例如：模拟 2 天"/></label><label>必要背景（选填）<input maxLength={2000} value={background} onChange={e => setBackground(e.target.value)} placeholder="与问题有关的模拟背景"/></label></div>
    <label className="check"><input type="checkbox" required checked={confirmed} onChange={e => setConfirmed(e.target.checked)}/>我确认仅提交公开或模拟信息，不包含真实敏感数据。</label>
    {error && <p role="alert" className="error">{error}</p>}<button disabled={busy || !confirmed} type="submit">{busy ? '正在提交…' : '提交并检索证据 →'}</button></form></section>
    <aside><section className="card sample"><p className="eyebrow">快速体验</p><h2>从一个模拟问题开始</h2><p>仅用于展示提交、检索和引用关联流程。</p><button className="secondary" onClick={() => {setSymptoms('模拟成年人出现轻微咳嗽，希望了解公开证据中的一般信息与局限。'); setDuration('模拟 2 天'); setBackground('无真实患者资料');}}>填入模拟示例 ↗</button></section><section className="notice"><h3>分析不是诊断</h3><p>结果可能不完整或不准确。证据不足时会明确弃答；引用关联不代表临床审核。真实健康问题请寻求专业医疗帮助。</p><a href="/assistant">原版助手仍可使用 →</a></section></aside></div></>;
}
function Task({id, result}: {id: string; result: boolean}) {
  const [analysis, setAnalysis] = useState<Analysis | null>(null); const [error, setError] = useState('');
  useEffect(() => { let stopped = false; let timer: ReturnType<typeof setTimeout>;
    const poll = async () => { try {
      const item = await api<Case>('/cases/' + id);
      if (['CREATED', 'REVISION_REQUIRED'].includes(item.status) && !item.active_task) await api('/cases/' + id + '/workflow', {method: 'POST', headers: {'Idempotency-Key': 'web-' + id}});
      const data = await api<Analysis>('/cases/' + id + '/analysis');
      if (!stopped) { setAnalysis(data); setError(''); if (data.outcome === 'processing') timer = setTimeout(poll, 1200); }
    } catch (e) { if (!stopped) setError((e as Error).message); } }; void poll();
    return () => { stopped = true; clearTimeout(timer); };
  }, [id]);
  return <><a className="back" href="#/history">← 咨询记录</a><p className="eyebrow">{result ? '03 / 结果与证据' : '02 / 任务进度'}</p><h1>{result ? '理解结论，也理解局限。' : '从问题到证据，逐步可见。'}</h1><p className="muted">任务 {id}</p>{error && <div role="alert" className="error">{error}<button onClick={() => location.reload()}>重新连接</button></div>}
    {!analysis ? <p role="status">正在读取任务…</p> : <><section className="card"><div className="section-title"><h2>{states[analysis.status] || analysis.status}</h2><span className="badge">{analysis.outcome === 'processing' ? '处理中' : '处理结束'}</span></div><p role="status">{analysis.message}</p>
    {!result && <><ol className="steps">{['提交', '检索', '分析', '审核', '结果'].map((label, i) => <li key={label}><b>0{i + 1}</b>{label}</li>)}</ol><p className="muted">可以刷新或关闭此页，再从咨询记录返回。任务由后端持续执行。</p></>}
    {analysis.outcome === 'processing' && <button className="secondary" onClick={async () => {try {await api('/cases/' + id + '/cancel', {method: 'POST'}); location.reload();} catch(e) {setError((e as Error).message);}}}>取消本次任务</button>}
    {analysis.outcome !== 'processing' && !result && <a className="button" href={'#/cases/' + id + '/result'}>查看结果与证据 →</a>}
    {analysis.outcome === 'failed' && <><p>错误标识：{analysis.failure_code || '需要维护者检查'}</p><a href="#/">重新提交新的咨询 →</a></>}
    </section>{result && <div className="columns"><section className="card"><h2>辅助分析</h2>{analysis.summary && <p>{analysis.summary}</p>}{analysis.claims.map(claim => <article key={claim.claim_id} className="claim"><p>{claim.text}</p>{claim.evidence_ids.map(e => <a className="citation" key={e} href={'#evidence-' + e} onClick={event => {event.preventDefault(); document.getElementById('evidence-' + e)?.scrollIntoView({behavior: 'smooth'});}}>查看证据 ↗</a>)}</article>)}<h3>不确定性与局限</h3><ul>{analysis.limitations.map((x, i) => <li key={i}>{x}</li>)}</ul><h3>风险与后续行动</h3><ul>{[...analysis.risk_warnings, ...analysis.next_steps].map((x, i) => <li key={i}>{x}</li>)}</ul><p className="notice">{analysis.disclaimer}</p></section><section className="card"><h2>证据来源 <span className="count">{analysis.evidence.length}</span></h2>{analysis.evidence.length === 0 && <p>没有可展示的证据，不作确定性结论。</p>}{analysis.evidence.map((e, i) => <article className="evidence" id={'evidence-' + e.chunk_id} key={e.chunk_id}><span className="eyebrow">证据 {i + 1} / {e.evidence_level}</span><p>{e.text}</p><p className="muted">{e.source} · {e.source_id}</p>{e.source_url ? <a href={e.source_url} target="_blank" rel="noreferrer">查看原始来源 ↗</a> : <small>本地模拟片段，无外部来源链接；不能作为临床证据。</small>}</article>)}</section></div>}</> }</>;
}
function History() {
  const [items, setItems] = useState<Case[]>([]); const [error, setError] = useState('');
  const [cursor, setCursor] = useState<number | null>(null); const [loading, setLoading] = useState(false);
  async function load(before?: number) {
    setLoading(true);
    try { const data = await api<{items: Case[]; next_cursor: number | null}>('/cases' + (before ? '?before=' + before : ''));
      setItems(previous => before ? [...previous, ...data.items] : data.items); setCursor(data.next_cursor); setError('');
    } catch {setError('暂时无法获取本人的历史记录，请检查连接。');} finally {setLoading(false);}
  }
  useEffect(() => {void load();}, []);
  return <><p className="eyebrow">04 / 咨询记录</p><h1>回看每一次证据探索。</h1><p className="muted">仅显示当前匿名会话的任务；会话有效期 30 天。清除 Cookie 或换浏览器后无法找回，不是实名账户。</p>{error && <p role="alert">{error}<button onClick={() => void load()}>重试</button></p>}<section className="card">{items.length ? items.map(item => <a className="history-item" key={item.case_id} href={'#/cases/' + item.case_id}><span>{item.case_id}</span><span>{states[item.status] || item.status} →</span></a>) : <p>{loading ? '正在读取…' : '暂无咨询记录。'}<a href="#/">开始第一条模拟咨询 →</a></p>}{cursor && <button disabled={loading} onClick={() => void load(cursor)}>加载更多</button>}</section></>;
}
// 先完成会话握手，避免首次并发请求被分配到不同匿名身份。
const root = createRoot(document.getElementById('root')!);
api('/session').then(() => root.render(<React.StrictMode><App/></React.StrictMode>)).catch(() => root.render(<main><h1>暂时无法建立会话</h1><p>请确认后端已启动；若会话过期，请刷新重建。旧会话记录不会自动归属于新会话。</p><button onClick={() => location.reload()}>刷新重试</button></main>));
