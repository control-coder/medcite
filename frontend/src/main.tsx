import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

type Case = { case_id: string; status: string; question?: string | null; active_task: { status: string } | null };
type Citation = { chunk_id: string; quote: string; entailment: 'entailed' | 'partial' };
type Evidence = { chunk_id: string; text: string; source: string; source_id: string; source_url: string | null; evidence_level: string };
type Analysis = { case_id: string; status: string; execution_mode: string; outcome: string; message: string; summary: string | null;
  observation: { recorded_stage_latency_ms: number; provider_attempts: number; recorded_input_tokens: number | null; recorded_output_tokens: number | null; note: string };
  answer_mode?: 'extractive' | 'generative';
  claims: { claim_id: string; text: string; evidence_ids: string[]; citations?: Citation[] }[];
  evidence: Evidence[];
  limitations: string[]; risk_warnings: string[]; next_steps: string[]; disclaimer: string; failure_code: string | null };
const modes: Record<string, string> = {
  fake_offline: '离线演示 · 固定样例',
  retrieval_mock: '离线演示 · 真实检索，模拟生成',
  mimo_grounded: '抽取式回答 · 由模型选择原文摘录',
  model_pipeline: '模型适配器链路 · 未经真实 API 验收',
  unknown: '运行模式未记录',
};
const GENERATIVE_MODE = '生成式回答 · 引用经逐字校验与模型蕴含判定';
const states: Record<string, string> = { CREATED: '排队中', NORMALIZED: '检索中', EVIDENCE_RETRIEVED: '检索完成', PLAN_GENERATED: '生成中', SPECIALIST_REVIEWING: '生成中', ARBITRATION_REVIEWING: '审核中', APPROVED: '整理结果', REPORT_GENERATED: '整理结果', CLOSED_SUCCESS: '已完成', ESCALATED: '处理受阻', REVISION_REQUIRED: '修订中', CLOSED_FAILED: '失败', CLOSED_CANCELLED: '已取消', CLOSED_ESCALATED: '已终止' };
// 进度步骤的纯展示映射：状态 → 五步中的位置；终态受阻/失败/取消返回 -1，不伪造进度。
const stepOrder: Record<string, number> = { CREATED: 0, NORMALIZED: 1, EVIDENCE_RETRIEVED: 1, PLAN_GENERATED: 2, SPECIALIST_REVIEWING: 2, REVISION_REQUIRED: 2, ARBITRATION_REVIEWING: 3, APPROVED: 4, REPORT_GENERATED: 4, CLOSED_SUCCESS: 5 };
const steps = ['提交', '检索资料', '生成回答', '审核引用', '完成'];
const examples = ['通风如何改善室内空气质量？', '登革热有哪些症状，怎样预防？', '流感和普通感冒有什么区别？'];
// 未填写持续时间时的默认值：科普提问通常没有病程，接口要求该字段非空。
const DEFAULT_DURATION = '不适用';

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch('/api/v1' + path, { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } });
  const data = await response.json();
  if (!response.ok) throw new Error(data.code === 'CASE_NOT_FOUND'
    ? '记录不存在或不属于当前匿名会话。清除 Cookie 或换浏览器后无法找回旧记录。'
    : typeof data.detail === 'string' ? data.detail : '输入不符合要求，请检查问题内容。');
  return data as T;
}
const go = (path: string) => { location.hash = path; };
// 咨询入口把表单拼成“症状：…\n持续时间：…\n背景：…”保存为 question，这里只取出问题本身用于展示。
const questionOf = (question?: string | null) => question ? (/^症状：([\s\S]*?)\n持续时间：/.exec(question)?.[1] || question) : '';
// 来源字段形如“世界卫生组织 | 标题”；没有分隔符时整体作为标题。
const splitSource = (source: string) => { const hit = /\s*[|｜]\s*/.exec(source); return hit && hit.index > 0 ? [source.slice(0, hit.index), source.slice(hit.index + hit[0].length)] : ['', source]; };
// 来源字段没有发布机构时，按原文链接的域名推断
const publishers: [RegExp, string][] = [[/(^|\.)who\.int$/, '世界卫生组织'], [/(^|\.)nhc\.gov\.cn$/, '国家卫生健康委'], [/(^|\.)chinacdc\.cn$/, '中国疾控中心'],
  [/(^|\.)gov\.cn$/, '中国政府网'], [/(^|\.)wikipedia\.org$/, '维基百科']];
const publisherOf = (url: string | null) => { try { const host = url ? new URL(url).hostname : ''; return publishers.find(([re]) => re.test(host))?.[1] || ''; } catch { return ''; } };
const isUrgent = (text: string) => text.includes('120');

function App() {
  const [route, setRoute] = useState(location.hash.slice(1) || '/');
  useEffect(() => { const change = () => setRoute(location.hash.slice(1) || '/'); window.addEventListener('hashchange', change); return () => window.removeEventListener('hashchange', change); }, []);
  const parts = route.split('/');
  return <>
    <header><div className="bar">
      <a className="brand" href="#/"><span className="brand-mark" aria-hidden="true">M</span>MedCite</a>
      <nav aria-label="主导航"><a href="#/" aria-current={route === '/' ? 'page' : undefined}>提问</a><a href="#/history" aria-current={route === '/history' ? 'page' : undefined}>历史记录</a></nav>
    </div></header>
    <main>{parts[1] === 'history' ? <History/> : parts[1] === 'cases' && parts[2] ? <Task key={parts[2]} id={parts[2]}/> : <Consult/>}</main>
    <footer>MedCite 工程原型 · 内容来自公开资料，仅供健康科普，不能替代医生诊断 · <a href="/demo">工程工作台</a></footer>
  </>;
}

function Consult() {
  const [symptoms, setSymptoms] = useState(''); const [duration, setDuration] = useState('');
  const [background, setBackground] = useState(''); const [kind, setKind] = useState('deidentified_simulation');
  const [source, setSource] = useState(''); const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const submit = async (event: React.FormEvent) => {
    event.preventDefault(); setBusy(true); setError('');
    const payload = {symptoms, duration: duration.trim() || DEFAULT_DURATION, background, input_kind: kind, source_ref: source || null, non_sensitive_confirmed: confirmed};
    // 只保存幂等键；不在浏览器持久保存问题与背景。
    const key = sessionStorage.getItem('medidiag-submit-key') || crypto.randomUUID(); sessionStorage.setItem('medidiag-submit-key', key);
    try { const item = await api<Case>('/consultations', {method: 'POST', headers: {'Idempotency-Key': key}, body: JSON.stringify(payload)}); sessionStorage.removeItem('medidiag-submit-key'); go('/cases/' + item.case_id); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  return <section className="ask">
    <h1>健康科普问答</h1>
    <p className="sub">回答来自世界卫生组织、国家卫生健康委、中国疾控中心等公开资料，每句话都标注出处；资料不足时不作答。</p>
    <form className="card" onSubmit={submit}>
      <label className="field">你的问题<textarea required minLength={10} maxLength={8000} value={symptoms} onChange={e => setSymptoms(e.target.value)} placeholder="例如：流感和普通感冒有什么区别？（至少 10 个字）" rows={4}/></label>
      <div className="examples"><span>试试：</span>{examples.map(text => <button type="button" className="chip" key={text} onClick={() => setSymptoms(text)}>{text}</button>)}</div>
      <details className="more"><summary>补充信息（选填）</summary>
        <div className="form-row"><label className="field">持续时间<input maxLength={200} value={duration} onChange={e => setDuration(e.target.value)} placeholder="例如：2 天；科普问题可不填"/></label>
          <label className="field">相关背景<input maxLength={2000} value={background} onChange={e => setBackground(e.target.value)} placeholder="与问题有关的背景"/></label></div>
        <div className="form-row"><label className="field">输入来源<select value={kind} onChange={e => setKind(e.target.value)}><option value="deidentified_simulation">模拟或脱敏输入</option><option value="public_dataset">公开数据集</option></select></label>
          {kind === 'public_dataset' && <label className="field">数据集标识<input required maxLength={128} value={source} onChange={e => setSource(e.target.value)} placeholder="数据集名称与样本编号"/></label>}</div>
      </details>
      <label className="check"><input type="checkbox" required checked={confirmed} onChange={e => setConfirmed(e.target.checked)}/>问题中不含姓名、证件号等真实个人信息</label>
      {error && <p role="alert" className="error">{error}</p>}
      <div className="form-foot"><button disabled={busy || !confirmed} type="submit">{busy ? '正在提交…' : '提交问题'}</button></div>
    </form>
    <p className="hint">本站不提供诊断、用药剂量或停药建议。出现急症请立即拨打 120。</p>
  </section>;
}

// 在证据原文中高亮被引用的 quote；比较时忽略空白，找不到时原样显示全文。
function Quoted({text, quote}: {text: string; quote: string | null}) {
  if (!quote) return <>{text}</>;
  const pattern = Array.from(quote.replace(/\s+/g, '')).map(ch => ch.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\s*');
  const hit = new RegExp(pattern, 'i').exec(text);
  if (!hit) return <>{text}</>;
  return <>{text.slice(0, hit.index)}<mark className="quote-hit">{hit[0]}</mark>{text.slice(hit.index + hit[0].length)}</>;
}

function Task({id}: {id: string}) {
  const [analysis, setAnalysis] = useState<Analysis | null>(null); const [question, setQuestion] = useState('');
  const [error, setError] = useState('');
  const [cited, setCited] = useState<{chunk_id: string; quote: string | null} | null>(null);
  const jump = (chunkId: string) => { const target = document.getElementById('evidence-' + chunkId); target?.scrollIntoView({behavior: 'smooth', block: 'center'}); target?.classList.add('flash'); setTimeout(() => target?.classList.remove('flash'), 1800); };
  useEffect(() => { let stopped = false; let timer: ReturnType<typeof setTimeout>; let source: EventSource | undefined;
    // 读取一次最新状态；返回是否仍在处理中（出错时视为停止）
    const load = async (): Promise<boolean> => { try {
      const item = await api<Case>('/cases/' + id);
      if (['CREATED', 'REVISION_REQUIRED'].includes(item.status) && !item.active_task) await api('/cases/' + id + '/workflow', {method: 'POST', headers: {'Idempotency-Key': 'web-' + id}});
      const data = await api<Analysis>('/cases/' + id + '/analysis');
      if (stopped) return false;
      setQuestion(questionOf(item.question)); setAnalysis(data); setError(''); return data.outcome === 'processing';
    } catch (e) { if (!stopped) setError((e as Error).message); return false; } };
    const poll = async () => { if (await load() && !stopped) timer = setTimeout(poll, 1200); };
    // 优先订阅 SSE（后端推送阶段变化）；不支持或连接出错时回退到轮询
    const listen = async () => {
      if (!(await load()) || stopped) return;
      if (typeof EventSource === 'undefined') { timer = setTimeout(poll, 1200); return; }
      const es = new EventSource('/api/v1/cases/' + id + '/events/stream'); source = es;
      let busy = false; let again = false;
      const refresh = async () => { if (busy) { again = true; return; } busy = true;
        do { again = false; if (!(await load())) { es.close(); break; } } while (again && !stopped); busy = false; };
      es.addEventListener('case_event', () => { void refresh(); });
      es.addEventListener('end', () => { es.close(); void poll(); });
      es.onerror = () => { es.close(); if (!stopped) timer = setTimeout(poll, 1200); };
    };
    void listen();
    return () => { stopped = true; clearTimeout(timer); source?.close(); };
  }, [id]);
  const done = analysis && (analysis.outcome === 'ready' || analysis.outcome === 'insufficient_evidence');
  return <section className="task">
    <a className="back" href="#/history">← 历史记录</a>
    <h1 className="question">{question || '问题'}</h1>
    {analysis && <p className="task-meta"><span className={'pill ' + statusTone(analysis.status)}>{states[analysis.status] || analysis.status}</span><span className="task-id">{id}</span></p>}
    {error && <div role="alert" className="error">{error}<button className="secondary" onClick={() => location.reload()}>重新连接</button></div>}
    {!analysis ? (!error && <p role="status" className="muted">正在读取…</p>)
      : done ? <Result analysis={analysis} cited={cited} onCite={(chunkId, quote) => { setCited({chunk_id: chunkId, quote}); jump(chunkId); }}/>
      : <section className="card progress">
        <h2>{states[analysis.status] || analysis.status}</h2>
        {analysis.outcome === 'processing' && <ol className="steps" style={{'--p': Math.max(0, Math.min(4, stepOrder[analysis.status] ?? -1)) / 4} as React.CSSProperties}>{steps.map((label, i) => { const step = stepOrder[analysis.status] ?? -1; return <li key={label} className={i < step ? 'done' : i === step ? 'current' : undefined}><b>{i < step ? '✓' : i + 1}</b>{label}</li>; })}</ol>}
        <p role="status">{analysis.outcome === 'processing' ? '正在检索资料并生成回答，通常需要半分钟到两分钟。可以离开此页，稍后在历史记录中查看。' : analysis.message}</p>
        {analysis.outcome !== 'processing' && <p className="muted">任务已结束，可从历史记录返回；刷新不会重新执行。</p>}
        {analysis.outcome === 'failed' && <p className="muted">错误标识：{analysis.failure_code || '需要维护者检查'}</p>}
        <div className="actions">{analysis.outcome === 'processing'
          ? <button className="secondary" onClick={async () => {try {await api('/cases/' + id + '/cancel', {method: 'POST'}); location.reload();} catch(e) {setError((e as Error).message);}}}>取消本次任务</button>
          : <a className="button" href="#/">重新提问</a>}</div>
      </section>}
  </section>;
}

function Result({analysis, cited, onCite}: {analysis: Analysis; cited: {chunk_id: string; quote: string | null} | null; onCite: (chunkId: string, quote: string | null) => void}) {
  const number = (chunkId: string) => analysis.evidence.findIndex(e => e.chunk_id === chunkId) + 1;
  const cite = (chunkId: string, quote: string | null, key: string) => { const n = number(chunkId);
    return <button type="button" className="cite-ref" key={key} aria-label={'引用 ' + n + '：跳转到来源 ' + n} onClick={() => onCite(chunkId, quote)}>{n}</button>; };
  const urgent = analysis.risk_warnings.filter(isUrgent);
  const notes = [...new Set([...analysis.limitations, ...analysis.risk_warnings.filter(x => !isUrgent(x)), ...analysis.next_steps])];
  const generative = analysis.answer_mode === 'generative';
  const ready = analysis.outcome === 'ready';
  const o = analysis.observation;
  const run = [...(ready ? [generative ? GENERATIVE_MODE : modes[analysis.execution_mode] || modes.unknown] : []),
    '处理耗时 ' + (o.recorded_stage_latency_ms / 1000).toFixed(1) + ' 秒', '阶段调用 ' + o.provider_attempts + ' 次',
    o.recorded_input_tokens === null ? 'Token 未记录' : 'Token ' + o.recorded_input_tokens.toLocaleString() + ' / ' + (o.recorded_output_tokens ?? 0).toLocaleString()];
  const sources = analysis.evidence.length > 0;
  return <div className={sources ? 'columns' : 'columns columns--single'}>
    <section className="card answer">
      <div className="card-head"><h2>回答</h2>{ready && <span className="tag">{generative ? '生成式回答' : '原文摘录'}</span>}</div>
      {urgent.map((x, i) => <p key={i} className="alert alert--urgent" role="alert">{x}</p>)}
      {!ready ? <div className={'abstain' + (isUrgent(analysis.message) ? ' abstain--urgent' : '')}><strong>{isUrgent(analysis.message) ? '请立即就医' : '暂时无法回答'}</strong><p>{analysis.message}</p></div>
        : generative ? <p className="answer-text">{analysis.claims.map(claim => <span className="claim" key={claim.claim_id}>{claim.text}{claim.citations?.some(c => c.entailment === 'partial') && <span className="partial-mark" title="模型判定引用只部分支持这句话">部分支持</span>}{(claim.citations?.length ? claim.citations.map(c => [c.chunk_id, c.quote] as const) : claim.evidence_ids.map(e => [e, null] as const)).map(([chunkId, quote]) => cite(chunkId, quote, chunkId + (quote || '')))}</span>)}</p>
        : <div className="excerpts">{analysis.claims.map(claim => <blockquote className="claim" key={claim.claim_id}>{claim.text}{claim.evidence_ids.map(e => cite(e, null, e))}</blockquote>)}</div>}
      {notes.length > 0 && <details className="notes"><summary>注意事项与局限（{notes.length}）</summary><ul>{notes.map((x, i) => <li key={i}>{x}</li>)}</ul></details>}
      <p className="run-meta" title={o.note}>{run.join(' · ')}</p>
      <p className="disclaimer">{analysis.disclaimer}{ready && ' 引用说明回答有原文支持，不代表医学审核。'}</p>
    </section>
    {sources && <section className="card sources">
      <div className="card-head"><h2>来源</h2><span className="count">{analysis.evidence.length}</span></div>
      {analysis.evidence.map((e, i) => { const [publisher, title] = splitSource(e.source);
        return <article className={'evidence' + (cited?.chunk_id === e.chunk_id ? ' evidence--active' : '')} id={'evidence-' + e.chunk_id} key={e.chunk_id}>
          <div className="evidence-head"><span className="num">{i + 1}</span><span className="publisher">{publisher || publisherOf(e.source_url) || '公开资料'}</span>{e.evidence_level === 'level_1_guideline' && <span className="tag tag--soft">指南</span>}</div>
          <p className="evidence-title">{title}</p>
          <p className="evidence-text"><Quoted text={e.text} quote={cited?.chunk_id === e.chunk_id ? cited.quote : null}/></p>
          {e.source_url && <a href={e.source_url} target="_blank" rel="noreferrer">查看原文 ↗</a>}
        </article>; })}
    </section>}
  </div>;
}

// 状态胶囊的纯展示语义分色：成功绿 / 进行中琥珀 / 失败终止红 / 取消灰。
const statusTone = (s: string) => s === 'CLOSED_SUCCESS' ? 'pill--ok' : s === 'CLOSED_CANCELLED' ? 'pill--muted' : s === 'CLOSED_FAILED' || s === 'CLOSED_ESCALATED' || s === 'ESCALATED' ? 'pill--bad' : 'pill--busy';
function History() {
  const [items, setItems] = useState<Case[]>([]); const [error, setError] = useState('');
  const [cursor, setCursor] = useState<number | null>(null); const [loading, setLoading] = useState(false);
  async function load(before?: number) {
    setLoading(true);
    try { const data = await api<{items: Case[]; next_cursor: number | null}>('/cases' + (before ? '?before=' + before : ''));
      setItems(previous => before ? [...previous, ...data.items] : data.items); setCursor(data.next_cursor); setError('');
    } catch {setError('暂时无法获取历史记录，请检查连接。');} finally {setLoading(false);}
  }
  useEffect(() => {void load();}, []);
  return <section className="history">
    <h1>历史记录</h1>
    <p className="sub">记录保存在当前浏览器的匿名会话中，有效期 30 天；清除 Cookie 或更换浏览器后无法找回。</p>
    {error && <p role="alert" className="error">{error}<button className="secondary" onClick={() => void load()}>重试</button></p>}
    <section className="card list">{items.length ? items.map(item => <a className="history-item" key={item.case_id} href={'#/cases/' + item.case_id}>
        <span className="history-text"><span className="history-q">{questionOf(item.question) || '等待处理的问题'}</span><span className="task-id">{item.case_id}</span></span>
        <span className={'pill ' + statusTone(item.status)}>{states[item.status] || item.status}</span></a>)
      : <p className="empty-state">{loading ? '正在读取…' : '暂无记录。'}<a href="#/">去提问 →</a></p>}
      {cursor && <button className="secondary more-btn" disabled={loading} onClick={() => void load(cursor)}>加载更多</button>}</section>
  </section>;
}

// 先完成会话握手，避免首次并发请求被分配到不同匿名身份。
const root = createRoot(document.getElementById('root')!);
api('/session').then(() => root.render(<React.StrictMode><App/></React.StrictMode>)).catch(() => root.render(<main><h1>暂时无法建立会话</h1><p>请确认后端已启动；若会话过期，请刷新重建。旧会话记录不会自动归属于新会话。</p><button onClick={() => location.reload()}>刷新重试</button></main>));
