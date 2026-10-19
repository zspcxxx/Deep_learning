const META = JSON.parse(document.getElementById("lesson-meta").textContent);
const $ = (id) => document.getElementById(id);

const VERDICT = {
  passed: "实验的通过条件全部满足。下面逐条解读结果中的关键数字。",
  failed: "脚本正常结束，但实验的通过条件没有全部满足。对照下面的解读和日志找出是哪一项没达标。",
  error: "脚本异常退出，没有产出结果。请看日志末尾的 Python 异常栈。",
  cancelled: "运行被手动停止，没有产出结果。",
};
const LEVEL_ICON = { info: "ℹ", ok: "✓", warn: "!", error: "✗" };

let es = null;
let timerHandle = null;
let lineNo = 0;
const lineEls = new Map(); // event index -> console line element

function resetView() {
  lineNo = 0;
  lineEls.clear();
  $("console").replaceChildren();
  $("notes").replaceChildren();
  $("result-panel").hidden = true;
  document.querySelectorAll("#steps li").forEach((li) => (li.className = ""));
}

function setStatus(status) {
  setBadge($("status"), status);
  const running = status === "running";
  $("run-btn").hidden = running;
  $("stop-btn").hidden = !running;
}

function startTimer(startedAt) {
  stopTimer();
  const t0 = startedAt ? new Date(startedAt).getTime() : Date.now();
  const tick = () => ($("timer").textContent = `${((Date.now() - t0) / 1000).toFixed(0)} s`);
  tick();
  timerHandle = setInterval(tick, 500);
}

function stopTimer(elapsed) {
  if (timerHandle) clearInterval(timerHandle);
  timerHandle = null;
  if (elapsed !== undefined && elapsed !== null) $("timer").textContent = `${Number(elapsed).toFixed(1)} s`;
}

function follow(el) {
  if ($("follow").checked) el.scrollTop = el.scrollHeight;
}

function markStep(index) {
  document.querySelectorAll("#steps li").forEach((li) => {
    const k = Number(li.dataset.step);
    li.className = k < index ? "done" : k === index ? "active" : li.className === "done" ? "done" : "";
  });
}

function focusLine(i) {
  const el = lineEls.get(i);
  if (!el) return;
  document.querySelectorAll(".console .line.hl").forEach((x) => x.classList.remove("hl"));
  el.classList.add("hl");
  el.scrollIntoView({ block: "center", behavior: "smooth" });
}

function handleEvent(ev) {
  if (ev.type === "log") {
    lineNo += 1;
    const line = h("div", { class: `line ${ev.level || ""}` }, h("span", { class: "ln", text: String(lineNo) }), h("span", { class: "tx", text: ev.text }));
    lineEls.set(ev.i, line);
    $("console").append(line);
    follow($("console"));
  } else if (ev.type === "note") {
    const lineEl = lineEls.get(ev.line);
    if (lineEl) lineEl.classList.add("annotated");
    const n = lineEl ? lineEl.querySelector(".ln").textContent : "";
    const card = h("div", { class: `note ${ev.level}`, onclick: () => focusLine(ev.line) },
      h("div", { class: "note-head" },
        h("span", { class: "icon", text: LEVEL_ICON[ev.level] || "ℹ" }),
        h("b", { text: ev.title }),
        n ? h("span", { class: "ref", text: `第 ${n} 行` }) : null),
      ev.body ? h("p", { text: ev.body }) : null);
    $("notes").append(card);
    follow($("notes"));
  } else if (ev.type === "step") {
    markStep(ev.index);
  }
}

function renderResult(rec) {
  setStatus(rec.status);
  stopTimer(rec.elapsed_sec);
  $("run-info").textContent = rec.finished_at ? `运行于 ${fmtTime(rec.started_at)}，耗时 ${Number(rec.elapsed_sec).toFixed(1)} s` : "";
  if (rec.status === "passed") markStep(META.steps.length);

  $("result-panel").hidden = false;
  $("verdict").replaceChildren(...[
    h("span", { class: "badge big", dataset: { status: rec.status }, text: STATUS_TEXT[rec.status] || rec.status }),
    h("span", { text: VERDICT[rec.status] || "" }),
    rec.status === "error" ? h("code", { text: `returncode=${rec.returncode}` }) : null,
  ].filter(Boolean));
  $("verdict").dataset.status = rec.status;

  const findings = rec.findings || [];
  $("findings").replaceChildren(...(findings.length ? findings.map((f) =>
    h("div", { class: `finding ${f.level}` },
      h("div", { class: "note-head" }, h("span", { class: "icon", text: LEVEL_ICON[f.level] || "ℹ" }), h("b", { text: f.title })),
      h("p", { text: f.body }))) : [h("p", { class: "placeholder", text: "没有可解读的结果。" })]));

  const v = encodeURIComponent(rec.finished_at || "");
  const images = rec.images || [];
  $("figures").replaceChildren(...(images.length ? images.map((img) =>
    h("figure", {},
      h("a", { href: `/file/${img.path}?v=${v}`, target: "_blank" }, h("img", { src: `/file/${img.path}?v=${v}`, alt: img.path, loading: "lazy" })),
      h("figcaption", {}, h("code", { text: img.path.split("/").pop() }), h("p", { text: img.note || "" })))) : [h("p", { class: "placeholder", text: "本次运行没有生成图片。" })]));

  $("raw-json").textContent = rec.result ? JSON.stringify(rec.result, null, 2) : "（无）";
}

function attach(run) {
  if (es) es.close();
  setStatus("running");
  startTimer(run.started_at);
  $("run-info").textContent = `运行中 · 开始于 ${fmtTime(run.started_at)}`;
  es = new EventSource(`/api/runs/${run.id}/events?since=-1`);
  es.onmessage = async (e) => {
    const ev = JSON.parse(e.data);
    handleEvent(ev);
    if (ev.type === "done") {
      es.close();
      es = null;
      const { body } = await api("GET", `/api/runs/${run.id}`);
      if (body) renderResult(body);
    }
  };
  es.onerror = async () => {
    // Server restarted or the run is gone: fall back to the persisted record.
    if (es && es.readyState === EventSource.CLOSED) {
      es = null;
      await loadLatest();
    }
  };
}

async function loadLatest() {
  const { body } = await api("GET", `/api/lessons/${encodeURIComponent(META.id)}/latest`);
  if (!body) {
    setStatus("none");
    return null;
  }
  resetView();
  if (body.status === "running") {
    attach(body);
  } else {
    (body.events || []).forEach(handleEvent);
    renderResult(body);
  }
  return body;
}

async function start() {
  $("run-btn").disabled = true;
  const { status, body } = await api("POST", `/api/lessons/${encodeURIComponent(META.id)}/run`);
  $("run-btn").disabled = false;
  if (status === 202) {
    resetView();
    attach(body);
  } else if (status === 409 && body && body.active) {
    if (body.active.lesson_id === META.id) {
      await loadLatest();
    } else if (confirm(`实验 ${body.active.lesson_id}「${body.active.title}」正在运行，同一时间只能运行一个。\n要打开它看进度吗？`)) {
      location.href = `/lesson/${body.active.lesson_id}`;
    }
  } else {
    alert(`启动失败（HTTP ${status}）`);
  }
}

$("run-btn").addEventListener("click", start);
$("stop-btn").addEventListener("click", async () => {
  const { body } = await api("GET", "/api/active");
  if (body && body.lesson_id === META.id) await api("POST", `/api/runs/${body.id}/cancel`);
});

(async () => {
  const latest = await loadLatest();
  if (window.AUTORUN && !(latest && latest.status === "running")) start();
})();
