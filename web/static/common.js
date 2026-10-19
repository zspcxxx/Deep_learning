const STATUS_TEXT = {
  running: "运行中", passed: "通过", failed: "未通过", error: "出错", cancelled: "已停止", none: "未运行",
};

function setBadge(el, status) {
  const s = status || "none";
  el.dataset.status = s;
  el.textContent = STATUS_TEXT[s] || s;
}

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else el.setAttribute(k, v);
  }
  for (const c of children) if (c !== null && c !== undefined) el.append(c);
  return el;
}

async function api(method, url) {
  const r = await fetch(url, { method, headers: { Accept: "application/json" } });
  let body = null;
  try { body = await r.json(); } catch (_) { /* empty body */ }
  return { ok: r.ok, status: r.status, body };
}

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString("zh-CN", { hour12: false });
}

async function pollActive() {
  const el = document.getElementById("active-run");
  if (!el) return;
  const { body } = await api("GET", "/api/active").catch(() => ({ body: null }));
  el.replaceChildren();
  if (body) {
    el.append(h("a", { href: `/lesson/${body.lesson_id}`, text: `● 正在运行：${body.lesson_id} ${body.title}` }));
    el.hidden = false;
  } else {
    el.hidden = true;
  }
}

document.querySelectorAll(".badge[data-status]").forEach((el) => setBadge(el, el.dataset.status));
pollActive();
setInterval(pollActive, 3000);
