function renderStats(lessons) {
  const el = document.getElementById("stats");
  const count = (s) => lessons.filter((l) => l.status === s).length;
  const items = [
    ["实验数", lessons.length],
    ["通过", count("passed")],
    ["未通过 / 出错", count("failed") + count("error")],
    ["未运行", lessons.filter((l) => !l.status).length],
  ];
  el.replaceChildren(...items.map(([k, v]) => h("div", { class: "stat" }, h("b", { text: String(v) }), h("span", { text: k }))));
}

function renderCards(lessons) {
  for (const l of lessons) {
    const card = document.querySelector(`.lesson-card[data-id="${l.id}"]`);
    if (!card) continue;
    setBadge(card.querySelector(".badge"), l.status);
    const parts = [];
    if (l.elapsed_sec) parts.push(`上次耗时 ${Number(l.elapsed_sec).toFixed(1)} s`);
    if (l.finished_at) parts.push(fmtTime(l.finished_at));
    card.querySelector("[data-meta]").textContent = parts.join(" · ");
    card.querySelector("[data-run]").disabled = l.status === "running";
  }
}

async function refresh() {
  const { ok, body } = await api("GET", "/api/lessons");
  if (!ok) return;
  renderStats(body);
  renderCards(body);
}

document.querySelectorAll("[data-run]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const id = btn.dataset.run;
    btn.disabled = true;
    const { status, body } = await api("POST", `/api/lessons/${id}/run`);
    if (status === 202) {
      location.href = `/lesson/${id}`;
      return;
    }
    btn.disabled = false;
    if (status === 409 && body && body.active) {
      if (confirm(`实验 ${body.active.lesson_id}「${body.active.title}」正在运行，同一时间只能运行一个。\n要打开它看进度吗？`)) {
        location.href = `/lesson/${body.active.lesson_id}`;
      }
    } else {
      alert(`启动失败（HTTP ${status}）`);
    }
  });
});

refresh();
setInterval(refresh, 3000);
