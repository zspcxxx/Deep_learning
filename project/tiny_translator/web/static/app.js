"use strict";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const STATUS = { preparing: "准备中", running: "训练中", stopping: "停止中", finished: "已完成", stopped: "已停止",
  failed: "失败", interrupted: "已中断", unknown: "未知" };
const COLORS = { train: "#2563eb", val: "#f97316", bleu: "#16a34a", lr: "#7c3aed" };
const EXAMPLES = ["I love you.", "How old are you?", "Where is the train station?", "It is raining today.",
  "I don't know what to do.", "She is reading a book in the library.", "Can you help me?", "Tom likes playing tennis."];

// The page may be mounted under a prefix (e.g. /translator/ inside the lesson dashboard).
const BASE = (document.body.dataset.base || "/").replace(/\/$/, "");

async function api(path, options = {}) {
  path = BASE + path;
  const init = { ...options };
  if (init.body && typeof init.body !== "string") {
    init.body = JSON.stringify(init.body);
    init.headers = { "Content-Type": "application/json" };
  }
  const resp = await fetch(path, init);
  const data = await resp.json().catch(() => ({ ok: false, error: `HTTP ${resp.status}` }));
  if (!resp.ok || data.ok === false) {
    const err = new Error(data.error || `HTTP ${resp.status}`);
    err.data = data;
    throw err;
  }
  return data;
}

function fmtDuration(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return "—";
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}时${m}分` : m ? `${m}分${s}秒` : `${s}秒`;
}
const fmtInt = (n) => (n === null || n === undefined ? "—" : Number(n).toLocaleString());
const fmtNum = (n, d = 2) => (n === null || n === undefined ? "—" : Number(n).toFixed(d));
const stat = (k, v) => `<div class="stat"><div class="k">${esc(k)}</div><div class="v">${v}</div></div>`;
const statusBadge = (s) => `<span class="status ${esc(s)}">${esc(STATUS[s] || s)}</span>`;

function setMsg(el, text, kind = "") {
  el.textContent = text || "";
  el.className = "hint" + (kind ? " " + kind : "");
}

// ================================================================ tabs

const TABS = ["config", "train", "eval", "guide"];
let currentTab = "config";

function showTab(name, { scrollId = null, replaceHash = true } = {}) {
  if (name && name.startsWith("guide-")) {
    scrollId = name;
    name = "guide";
  }
  if (!TABS.includes(name)) name = "config";
  currentTab = name;
  for (const t of TABS) $(`#tab-${t}`).hidden = t !== name;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  if (replaceHash) {
    const hash = scrollId || name;
    if (location.hash !== `#${hash}`) history.replaceState(null, "", `#${hash}`);
  }
  if (name === "train") renderTrain();
  if (name === "eval") { loadRuns(); renderEvalCharts(); }
  if (name === "guide" && scrollId) {
    requestAnimationFrame(() => {
      const el = document.getElementById(scrollId);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }
}

function applyHash() {
  const h = location.hash.slice(1);
  if (!h) { showTab("config", { replaceHash: false }); return; }
  if (h === "guide" || h.startsWith("guide-")) showTab(h, { replaceHash: false });
  else showTab(h, { replaceHash: false });
}

document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.querySelectorAll("[data-goto-guide]").forEach((b) => b.addEventListener("click", () => showTab("guide")));
window.addEventListener("hashchange", applyHash);
window.addEventListener("resize", () => { if (currentTab === "train") drawTrainCharts(); if (currentTab === "eval") renderEvalCharts(); });

// In-page guide TOC: keep the guide tab and update the hash.
document.addEventListener("click", (e) => {
  const a = e.target.closest('#tab-guide a[href^="#guide"]');
  if (!a) return;
  e.preventDefault();
  showTab(a.getAttribute("href").slice(1));
});

// ================================================================ config

const cfgState = { schema: null, datasets: [] };
const FORM_KEY = "tiny-translator-config";

async function loadConfig() {
  cfgState.datasets = await api("/api/datasets");
  cfgState.schema = await api("/api/config/schema");
  renderDatasets();
  renderPresets();
  renderForm();
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(FORM_KEY) || "null"); } catch (_) { saved = null; }
  fillForm({ ...cfgState.schema.defaults, ...(saved || {}) });
}

function renderDatasets() {
  const box = $("#datasetList");
  box.innerHTML = cfgState.datasets.map((d) => `
    <div class="dataset">
      <span class="name">${esc(d.name)}</span>
      <span class="muted">${esc(d.label)}</span>
      ${d.ready
        ? `<span class="badge ok">已就绪 · ${fmtInt(d.pairs)} 句对</span>`
        : d.downloadable
          ? `<span class="badge warn">未下载</span><button class="btn small" data-download="${esc(d.name)}" type="button">下载</button>`
          : `<span class="badge">缺少文件</span>`}
    </div>`).join("") || `<p class="muted">data/ 下没有数据集。</p>`;
  box.querySelectorAll("[data-download]").forEach((b) => b.addEventListener("click", () => downloadDataset(b.dataset.download, b)));
}

async function downloadDataset(name, button) {
  button.disabled = true;
  button.textContent = "下载中…";
  try {
    const res = await api(`/api/datasets/${encodeURIComponent(name)}/download`, { method: "POST" });
    cfgState.datasets = res.datasets;
    renderDatasets();
    setMsg($("#configMsg"), `已下载 ${fmtInt(res.pairs)} 个句对`, "ok");
  } catch (e) {
    button.disabled = false;
    button.textContent = "重试下载";
    setMsg($("#configMsg"), e.message, "err");
  }
}

function renderPresets() {
  const box = $("#presets");
  box.innerHTML = '<span class="muted small">预设：</span>' + cfgState.schema.presets
    .map((p) => `<button class="btn small" type="button" data-preset="${esc(p.key)}">${esc(p.label)}</button>`).join("");
  box.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
    const preset = cfgState.schema.presets.find((p) => p.key === b.dataset.preset);
    fillForm({ ...cfgState.schema.defaults, dataset: readForm().dataset, ...preset.values });
    $("#checkCard").hidden = true;
    setMsg($("#configMsg"), `已套用预设：${preset.label}`);
  }));
}

function inputFor(f) {
  const id = `f_${f.key}`;
  if (f.kind === "bool") return `<input type="checkbox" id="${id}" data-key="${f.key}">`;
  if (f.kind === "choice") {
    return `<select id="${id}" data-key="${f.key}">${(f.choices || []).map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("")}</select>`;
  }
  if (f.kind === "str") return `<input type="text" id="${id}" data-key="${f.key}">`;
  const step = f.step ?? (f.kind === "int" ? 1 : "any");
  return `<input type="number" id="${id}" data-key="${f.key}" step="${step}"${f.min !== undefined ? ` min="${f.min}"` : ""}${f.max !== undefined ? ` max="${f.max}"` : ""}>`;
}

function renderForm() {
  const { groups, fields } = cfgState.schema;
  $("#configForm").innerHTML = groups.map(([gid, glabel]) => `
    <div class="group"><h3>${esc(glabel)}</h3>
      ${fields.filter((f) => f.group === gid).map((f) => `
        <div class="field ${f.kind === "bool" ? "bool" : ""} ${f.key === "dataset" ? "wide" : ""}" data-field="${f.key}">
          <label for="f_${f.key}">${esc(f.label)}</label>${inputFor(f)}
          ${f.help ? `<div class="help">${esc(f.help)}</div>` : ""}
        </div>`).join("")}
    </div>`).join("");
  $("#configForm").addEventListener("input", () => {
    localStorage.setItem(FORM_KEY, JSON.stringify(readForm()));
    clearFieldErrors();
  });
}

function fillForm(values) {
  for (const f of cfgState.schema.fields) {
    const el = $(`#f_${f.key}`);
    if (!el || values[f.key] === undefined) continue;
    if (f.kind === "bool") el.checked = !!values[f.key];
    else el.value = values[f.key];
  }
  localStorage.setItem(FORM_KEY, JSON.stringify(readForm()));
}

function readForm() {
  const out = {};
  for (const f of cfgState.schema.fields) {
    const el = $(`#f_${f.key}`);
    if (!el) continue;
    if (f.kind === "bool") out[f.key] = el.checked;
    else if (f.kind === "int" || f.kind === "float") out[f.key] = el.value === "" ? null : Number(el.value);
    else out[f.key] = el.value;
  }
  return out;
}

function clearFieldErrors() {
  document.querySelectorAll(".field.invalid").forEach((el) => {
    el.classList.remove("invalid");
    el.querySelector(".err")?.remove();
  });
}

function showFieldErrors(errors) {
  clearFieldErrors();
  for (const [key, msg] of Object.entries(errors || {})) {
    const box = document.querySelector(`[data-field="${key}"]`);
    if (!box) continue;
    box.classList.add("invalid");
    box.insertAdjacentHTML("beforeend", `<div class="err">${esc(msg)}</div>`);
  }
}

async function checkConfig() {
  const msg = $("#configMsg");
  setMsg(msg, "正在读取数据、构建词表和模型…");
  $("#btnCheck").disabled = true;
  try {
    const res = await api("/api/config/check", { method: "POST", body: readForm() });
    const s = res.stats;
    $("#checkResult").innerHTML = [
      stat("模型参数量", fmtInt(res.params)),
      stat("训练 / 验证 / 测试", `${fmtInt(s.train)} / ${fmtInt(s.val)} / ${fmtInt(s.test)}`),
      stat("英文词表 / 中文字表", `${fmtInt(s.src_vocab)} / ${fmtInt(s.tgt_vocab)}`),
      stat("每轮步数 / 总步数", `${fmtInt(res.steps_per_epoch)} / ${fmtInt(res.total_steps)}`),
      stat("平均长度（英词 / 中字）", `${s.avg_src_len} / ${s.avg_tgt_len}`),
      stat("长度过滤丢弃", fmtInt(s.dropped_by_length)),
      stat("验证集英文未登录词率", `${(100 * s.val_src_unk_rate).toFixed(2)}%`),
    ].join("");
    $("#checkCard").hidden = false;
    setMsg(msg, "配置有效，可以开始训练", "ok");
  } catch (e) {
    showFieldErrors(e.data?.fields);
    setMsg(msg, e.data?.need_download ? "数据集尚未下载：点上方“下载”，或直接开始训练（会自动下载）" : e.message, "err");
  } finally {
    $("#btnCheck").disabled = false;
  }
}

async function startTraining() {
  const msg = $("#configMsg");
  $("#btnStart").disabled = true;
  try {
    const res = await api("/api/train/start", { method: "POST", body: readForm() });
    setMsg(msg, `已开始：${res.run_id}`, "ok");
    resetTrain();
    showTab("train");
    pollTrain();
  } catch (e) {
    showFieldErrors(e.data?.fields);
    setMsg(msg, e.message, "err");
  } finally {
    $("#btnStart").disabled = false;
  }
}

$("#btnCheck").addEventListener("click", checkConfig);
$("#btnStart").addEventListener("click", startTraining);

// ================================================================ training

const train = { runId: null, next: 0, state: null, config: null, steps: [], epochs: [], logs: [], timer: null,
  gpuPeak: null };
const gpuLive = { last: null, busy: 0, timer: null, peak: null };

function resetTrain() {
  Object.assign(train, { runId: null, next: 0, state: null, config: null, steps: [], epochs: [], logs: [], gpuPeak: null });
}

function applyGpu(gpu) {
  if (!gpu) return;
  gpuLive.last = gpu;
  if (typeof gpu.util === "number") {
    gpuLive.peak = gpuLive.peak === null ? gpu.util : Math.max(gpuLive.peak, gpu.util);
    if (train.state && isActive(train.state.status)) {
      train.gpuPeak = train.gpuPeak === null ? gpu.util : Math.max(train.gpuPeak, gpu.util);
    }
  }
  renderGpuMeter();
  if (train.state && isActive(train.state.status)) {
    renderJobPill();
    if (currentTab === "train") renderTrain(false);
  }
}

function renderGpuMeter() {
  const meter = $("#gpuMeter");
  const fill = $("#gpuFill");
  const pctEl = $("#gpuPct");
  if (!meter || !fill || !pctEl) return;
  const g = gpuLive.last;
  const util = g && typeof g.util === "number" ? g.util : null;
  const active = gpuLive.busy > 0 || (train.state && isActive(train.state.status));
  meter.classList.toggle("active", active && util !== null && util > 0);
  if (util === null) {
    fill.style.width = "0%";
    fill.className = "gpu-fill";
    pctEl.textContent = "—";
    meter.title = g?.name ? `${g.name}（暂无利用率读数）` : "GPU 使用率";
    return;
  }
  fill.style.width = `${util}%`;
  fill.className = "gpu-fill" + (util >= 70 ? " high" : util >= 35 ? " mid" : "");
  pctEl.textContent = `${util}%`;
  const mem = (g.mem_used_mb != null && g.mem_total_mb != null)
    ? ` · 显存 ${fmtInt(g.mem_used_mb)}/${fmtInt(g.mem_total_mb)} MB` : "";
  const peak = gpuLive.peak !== null ? ` · 峰值 ${gpuLive.peak}%` : "";
  meter.title = `${g.name || "GPU"}：${util}%${peak}${mem}`;
}

function beginGpuWatch() {
  gpuLive.busy += 1;
  scheduleGpuPoll(true);
}

function endGpuWatch() {
  gpuLive.busy = Math.max(0, gpuLive.busy - 1);
  scheduleGpuPoll(false);
}

async function pollGpuOnce() {
  try {
    const res = await api("/api/system/gpu");
    applyGpu(res.gpu);
  } catch (e) {
    console.warn("gpu poll failed", e);
  }
}

function scheduleGpuPoll(forceFast = false) {
  clearTimeout(gpuLive.timer);
  const active = forceFast || gpuLive.busy > 0 || (train.state && isActive(train.state.status));
  gpuLive.timer = setTimeout(async () => {
    // While training, /api/train/status already carries gpu; only hit /api/system/gpu for eval/translate
    // or when idle (slow keepalive so the meter stays fresh).
    if (gpuLive.busy > 0 || !(train.state && isActive(train.state.status))) {
      await pollGpuOnce();
    }
    scheduleGpuPoll(false);
  }, active ? 500 : 4000);
}

async function pollTrain() {
  clearTimeout(train.timer);
  try {
    let snap = await api(`/api/train/status?since=${train.next}`);
    if (snap.state && snap.state.run_id !== train.runId) {
      resetTrain();
      snap = await api("/api/train/status?since=0");
    }
    if (snap.gpu) applyGpu(snap.gpu);
    if (snap.state) {
      if (!train.runId) { train.runId = snap.state.run_id; train.config = snap.config; }
      const wasDone = train.state && !isActive(train.state.status);
      train.state = snap.state;
      for (const ev of snap.events) {
        if (ev.kind === "step") train.steps.push(ev);
        else if (ev.kind === "epoch") train.epochs.push(ev);
        else if (ev.kind === "log") train.logs.push(`${new Date(ev.t * 1000).toLocaleTimeString()}  ${ev.text}`);
        else if (ev.kind === "done" && !wasDone) loadRuns();
      }
      train.next = snap.next;
    }
    renderJobPill();
    if (currentTab === "train") renderTrain(snap.events.length > 0);
  } catch (e) {
    console.warn("status poll failed", e);
  }
  const active = train.state && isActive(train.state.status);
  train.timer = setTimeout(pollTrain, active ? 1000 : 5000);
}

const isActive = (s) => ["preparing", "running", "stopping"].includes(s);

function renderJobPill() {
  const pill = $("#jobPill");
  const s = train.state;
  if (!s) { pill.hidden = true; return; }
  pill.hidden = false;
  pill.classList.toggle("running", isActive(s.status));
  const pct = s.total_steps ? Math.round((100 * s.step) / s.total_steps) : 0;
  const gpu = gpuLive.last && typeof gpuLive.last.util === "number" ? ` · GPU ${gpuLive.last.util}%` : "";
  pill.textContent = isActive(s.status)
    ? `${s.name}：${STATUS[s.status]} ${pct}%${gpu}`
    : `${s.name}：${STATUS[s.status] || s.status}`;
  pill.onclick = () => showTab("train");
}

function renderTrain(changed = true) {
  const s = train.state;
  $("#trainEmpty").hidden = !!s;
  $("#trainBody").hidden = !s;
  if (!s) return;
  $("#trainTitle").textContent = `${s.name} · ${s.run_id}`;
  const st = s.stats;
  $("#trainMeta").textContent = [
    s.device && `设备 ${s.device}`,
    s.params && `参数 ${fmtInt(s.params)}`,
    st && `训练 ${fmtInt(st.train)} 句 · 词表 ${fmtInt(st.src_vocab)}/${fmtInt(st.tgt_vocab)}`,
    s.message,
  ].filter(Boolean).join("　·　");
  $("#trainStatus").outerHTML = `<span class="status ${s.status}" id="trainStatus">${STATUS[s.status] || s.status}</span>`;
  $("#btnStop").hidden = !["preparing", "running"].includes(s.status);
  $("#btnToEval").hidden = isActive(s.status) || !["finished", "stopped"].includes(s.status);

  const pct = s.total_steps ? (100 * s.step) / s.total_steps : 0;
  $("#trainBar").style.width = `${pct.toFixed(1)}%`;
  const last = train.steps[train.steps.length - 1];
  const eta = s.step && s.total_steps && isActive(s.status) ? (s.elapsed / s.step) * (s.total_steps - s.step) : null;
  const gpuNow = gpuLive.last && typeof gpuLive.last.util === "number" ? `${gpuLive.last.util}%` : "—";
  const gpuPeak = train.gpuPeak !== null ? `${train.gpuPeak}%` : "—";
  $("#trainStats").innerHTML = [
    stat("轮次", `${s.epoch} / ${s.epochs}`),
    stat("步数", `${fmtInt(s.step)} / ${fmtInt(s.total_steps)}`),
    stat("已用时间", fmtDuration(s.elapsed)),
    stat("预计剩余", fmtDuration(eta)),
    stat("当前训练损失", last ? fmtNum(last.loss, 3) : "—"),
    stat("吞吐（目标 token/s）", last ? fmtInt(last.tok_s) : "—"),
    stat("GPU 使用率", `${gpuNow}（峰值 ${gpuPeak}）`),
    stat("最佳验证 BLEU", s.best_val_bleu !== null ? `${fmtNum(s.best_val_bleu)}（第 ${s.best_epoch} 轮）` : "—"),
    stat("测试集 BLEU", s.test_bleu !== null ? fmtNum(s.test_bleu) : "—"),
  ].join("");

  if (!changed) return;
  drawTrainCharts();
  $("#epochTable tbody").innerHTML = train.epochs.slice().reverse().map((e) => `
    <tr class="${e.best ? "best" : ""}"><td>${e.epoch}${e.best ? " ★" : ""}</td><td class="num">${fmtNum(e.train_loss, 3)}</td>
    <td class="num">${fmtNum(e.val_loss, 3)}</td><td class="num">${fmtNum(e.val_ppl, 1)}</td>
    <td class="num">${fmtNum(e.val_bleu)}</td><td class="num">${e.seconds}s</td></tr>`).join("");
  const lastEpoch = train.epochs[train.epochs.length - 1];
  $("#sampleEpoch").textContent = lastEpoch ? `（第 ${lastEpoch.epoch} 轮）` : "";
  $("#samples").innerHTML = lastEpoch ? lastEpoch.samples.map((x) => `
    <div class="sample"><div class="en">${esc(x.en)}</div>
      <div class="zh"><span class="tag">参考</span><span>${esc(x.ref)}</span><span class="tag">模型</span><span>${esc(x.hyp) || '<i class="muted">（空）</i>'}</span></div>
    </div>`).join("") : '<p class="muted">第一轮结束后显示。</p>';
  const log = $("#trainLog");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 20;
  log.textContent = train.logs.join("\n");
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function drawTrainCharts() {
  if ($("#trainBody").hidden) return;
  lineChart($("#chartLoss"), {
    xLabel: "步", series: [
      { label: "训练损失", color: COLORS.train, points: train.steps.map((p) => [p.step, p.loss]), width: 1.2 },
      { label: "验证损失", color: COLORS.val, points: train.epochs.map((e) => [e.step, e.val_loss]), dots: true },
    ],
  });
  lineChart($("#chartBleu"), {
    xLabel: "轮", yMin: 0, series: [{ label: "验证 BLEU", color: COLORS.bleu, points: train.epochs.map((e) => [e.epoch, e.val_bleu]), dots: true }],
  });
  lineChart($("#chartLr"), {
    xLabel: "步", yMin: 0, series: [{ label: "学习率", color: COLORS.lr, points: train.steps.map((p) => [p.step, p.lr]) }],
  });
}

$("#btnStop").addEventListener("click", async () => {
  if (!confirm("确定停止训练？已训练的权重会保存，并在测试集上评估。")) return;
  try { await api("/api/train/stop", { method: "POST" }); pollTrain(); } catch (e) { alert(e.message); }
});
$("#btnToEval").addEventListener("click", () => { evalState.selected = train.runId; showTab("eval"); selectRun(train.runId); });

// ================================================================ evaluation

const evalState = { runs: [], selected: null, detail: null, lastEval: null };

async function loadRuns() {
  try { evalState.runs = await api("/api/runs"); } catch (e) { return; }
  const box = $("#runList");
  box.innerHTML = evalState.runs.map((r) => `
    <div class="run ${r.id === evalState.selected ? "active" : ""} ${r.has_checkpoint ? "" : "disabled"}" data-run="${esc(r.id)}">
      <div class="title"><span>${esc(r.name)}</span>${statusBadge(r.status)}</div>
      <div class="meta">${esc(r.id)}</div>
      <div class="meta">已训练 ${r.epochs_done}/${r.epochs ?? "?"} 轮 · 验证 BLEU ${fmtNum(r.best_val_bleu)} · 测试 BLEU ${fmtNum(r.test_bleu)}</div>
    </div>`).join("") || '<p class="muted">还没有训练记录。</p>';
  box.querySelectorAll(".run").forEach((el) => el.addEventListener("click", () => {
    const run = evalState.runs.find((r) => r.id === el.dataset.run);
    if (run && run.has_checkpoint) selectRun(run.id);
  }));
  if (!evalState.selected) {
    const first = evalState.runs.find((r) => r.has_checkpoint && !isActive(r.status));
    if (first) selectRun(first.id);
  }
}

async function selectRun(id) {
  evalState.selected = id;
  document.querySelectorAll(".run").forEach((el) => el.classList.toggle("active", el.dataset.run === id));
  let detail;
  try { detail = await api(`/api/runs/${encodeURIComponent(id)}`); } catch (e) { alert(e.message); return; }
  evalState.detail = detail;
  evalState.lastEval = null;
  $("#evalEmpty").hidden = true;
  $("#evalBody").hidden = false;
  $("#translateOut").hidden = true;
  $("#evalResult").hidden = true;
  setMsg($("#translateMsg"), "");
  setMsg($("#evalMsg"), "");

  const { config: c, summary: s, history: h } = detail;
  $("#evalTitle").textContent = c.name || id;
  $("#evalMeta").innerHTML = `${esc(id)}　${statusBadge(s.status || "unknown")}　${esc(s.device || "")}`;
  const secs = s.finished_at && s.started_at ? s.finished_at - s.started_at : null;
  $("#evalStats").innerHTML = [
    stat("最佳验证 BLEU", s.best_val_bleu !== undefined && s.best_val_bleu !== null ? `${fmtNum(s.best_val_bleu)}（第 ${s.best_epoch} 轮）` : "—"),
    stat("测试集 BLEU", fmtNum(s.test_bleu)),
    stat("参数量", fmtInt(s.params)),
    stat("结构", `${c.n_enc_layers}+${c.n_dec_layers} 层 · d=${c.d_model} · ${c.n_heads} 头`),
    stat("训练轮数", `${s.epochs_done ?? 0} / ${c.epochs}`),
    stat("训练用时", fmtDuration(secs)),
    s.stats ? stat("训练句对", fmtInt(s.stats.train)) : "",
  ].join("");
  $("#evalConfig").innerHTML = Object.entries(c).map(([k, v]) => `<div><span>${esc(k)}</span><span>${esc(v)}</span></div>`).join("");

  const fromHistory = (h.epochs.length ? h.epochs[h.epochs.length - 1].samples : []).map((x) => x.en).slice(0, 3);
  $("#examples").innerHTML = '<span class="muted small">试试：</span>' + [...EXAMPLES, ...fromHistory]
    .map((t) => `<span class="chip">${esc(t)}</span>`).join("");
  $("#examples").querySelectorAll(".chip").forEach((el) => el.addEventListener("click", () => {
    $("#srcText").value = el.textContent;
    translate();
  }));
  renderEvalCharts();
}

function renderEvalCharts() {
  const d = evalState.detail;
  if (!d || $("#evalBody").hidden) return;
  const h = d.history;
  lineChart($("#evalChartLoss"), {
    xLabel: "步", series: [
      { label: "训练损失", color: COLORS.train, points: h.steps.map((p) => [p.step, p.loss]), width: 1.2 },
      { label: "验证损失", color: COLORS.val, points: h.epochs.map((e) => [e.step, e.val_loss]), dots: true },
    ],
  });
  lineChart($("#evalChartBleu"), {
    xLabel: "轮", yMin: 0, series: [{ label: "验证 BLEU", color: COLORS.bleu, points: h.epochs.map((e) => [e.epoch, e.val_bleu]), dots: true }],
  });
}

async function translate() {
  const text = $("#srcText").value.trim();
  const msg = $("#translateMsg");
  if (!text) { setMsg(msg, "请输入英文句子", "err"); return; }
  if (!evalState.selected) return;
  setMsg(msg, "翻译中…");
  $("#btnTranslate").disabled = true;
  beginGpuWatch();
  try {
    const res = await api(`/api/runs/${encodeURIComponent(evalState.selected)}/translate`, {
      method: "POST", body: { text, beam: Number($("#beamSize").value), checkpoint: $("#evalCkpt").value },
    });
    $("#translateOut").hidden = false;
    $("#outText").textContent = res.output || "（模型没有生成内容）";
    $("#srcTokens").innerHTML = res.src_tokens.map((t, i) => `<span class="tok ${res.src_unk[i] ? "unk" : ""}">${esc(t)}</span>`).join("");
    renderHeatmap(res);
    const unk = res.src_unk.filter(Boolean).length;
    const gpuNote = gpuLive.last && typeof gpuLive.last.util === "number" ? `；GPU ${gpuLive.last.util}%` : "";
    setMsg(msg, `用时 ${res.elapsed_ms} ms${unk ? `；${unk} 个词不在词表中` : ""}${gpuNote}`, unk ? "" : "ok");
  } catch (e) {
    setMsg(msg, e.message, "err");
  } finally {
    endGpuWatch();
    $("#btnTranslate").disabled = false;
  }
}

function renderHeatmap(res) {
  const cols = [...res.src_tokens, "</s>"];
  const rows = res.out_tokens;
  if (!rows.length) { $("#heatmap").innerHTML = ""; return; }
  const head = `<thead><tr><th></th>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>`;
  const body = rows.map((ch, i) => `<tr><th>${esc(ch)}</th>${(res.attention[i] || []).map((v, j) => {
    const a = Math.pow(Math.max(0, Math.min(1, v)), 0.7);
    return `<td style="background: rgba(37, 99, 235, ${a.toFixed(3)})" title="${esc(ch)} ← ${esc(cols[j])}: ${v.toFixed(3)}">${v.toFixed(2)}</td>`;
  }).join("")}</tr>`).join("");
  $("#heatmap").innerHTML = head + `<tbody>${body}</tbody>`;
}

async function evaluate() {
  if (!evalState.selected) return;
  const msg = $("#evalMsg");
  setMsg(msg, "评估中，长句较多时束搜索会慢一些…");
  $("#btnEvaluate").disabled = true;
  const peakBefore = gpuLive.peak;
  gpuLive.peak = null;
  beginGpuWatch();
  try {
    const res = await api(`/api/runs/${encodeURIComponent(evalState.selected)}/evaluate`, {
      method: "POST",
      body: { split: $("#evalSplit").value, limit: Number($("#evalLimit").value), beam: Number($("#evalBeam").value), checkpoint: $("#evalCkpt").value },
    });
    evalState.lastEval = res;
    $("#evalResult").hidden = false;
    const evalPeak = gpuLive.peak;
    $("#evalScore").innerHTML = [
      stat("BLEU（字级）", fmtNum(res.bleu)),
      stat("1/2/3/4-gram 精确率", res.precisions.map((p) => p.toFixed(1)).join(" / ")),
      stat("长度惩罚 BP", fmtNum(res.bp, 3)),
      stat("完全一致", `${fmtNum(res.exact_match, 1)}%`),
      stat("英文未登录词率", `${fmtNum(res.src_unk_rate, 1)}%`),
      stat("句数 / 用时", `${fmtInt(res.count)} / ${res.seconds}s`),
      stat("评估期 GPU 峰值", evalPeak !== null ? `${evalPeak}%` : "—"),
    ].join("");
    renderEvalTable();
    setMsg(msg, evalPeak !== null ? `评估完成 · GPU 峰值 ${evalPeak}%` : "评估完成", "ok");
    if (peakBefore !== null) gpuLive.peak = Math.max(peakBefore, evalPeak ?? 0);
  } catch (e) {
    setMsg(msg, e.message, "err");
    if (peakBefore !== null) gpuLive.peak = peakBefore;
  } finally {
    endGpuWatch();
    $("#btnEvaluate").disabled = false;
  }
}

function renderEvalTable() {
  const res = evalState.lastEval;
  if (!res) return;
  const rows = res.samples.map((s, i) => ({ ...s, i: i + 1 }));
  const sort = $("#evalSort").value;
  if (sort === "worst") rows.sort((a, b) => a.bleu - b.bleu);
  if (sort === "best") rows.sort((a, b) => b.bleu - a.bleu);
  $("#evalTable tbody").innerHTML = rows.map((s) => `
    <tr><td>${s.i}</td><td>${esc(s.en)}</td><td>${esc(s.ref)}</td><td>${esc(s.hyp)}</td><td class="num">${fmtNum(s.bleu, 1)}</td></tr>`).join("");
}

$("#btnTranslate").addEventListener("click", translate);
$("#srcText").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); translate(); } });
$("#btnEvaluate").addEventListener("click", evaluate);
$("#evalSort").addEventListener("change", renderEvalTable);
$("#btnRefreshRuns").addEventListener("click", loadRuns);
$("#evalCkpt").addEventListener("change", () => { $("#translateOut").hidden = true; $("#evalResult").hidden = true; });
$("#btnDeleteRun").addEventListener("click", async () => {
  const id = evalState.selected;
  if (!id || !confirm(`删除运行 ${id} 及其检查点？`)) return;
  try {
    await api(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
    evalState.selected = null;
    evalState.detail = null;
    $("#evalBody").hidden = true;
    $("#evalEmpty").hidden = false;
    loadRuns();
  } catch (e) { alert(e.message); }
});

// ================================================================ boot

loadConfig().catch((e) => setMsg($("#configMsg"), `加载配置失败：${e.message}`, "err"));
applyHash();
pollTrain();
scheduleGpuPoll(true);
