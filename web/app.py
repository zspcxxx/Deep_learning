#!/usr/bin/env python3
from __future__ import annotations

import json

from flask import Flask, Response, abort, render_template_string, send_from_directory

from dllab.catalog import LESSONS, find_lesson
from dllab.paths import ARTIFACTS_DIR, PROJECT_ROOT

app = Flask(__name__)
DOCS = PROJECT_ROOT / "docs"
ARCH_DOC = DOCS / "architecture.html"

PAGE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deep Learning Lab 验证结果</title>
<style>
:root { --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --ok:#34d399; --bad:#f87171; --run:#fbbf24; --accent:#818cf8; }
* { box-sizing:border-box; }
body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:linear-gradient(160deg,#0f172a,#1e1b4b); color:var(--text); }
header { padding:36px 24px 16px; text-align:center; }
h1 { margin:0 0 8px; font-size:2rem; }
.sub { color:var(--muted); }
.wrap { max-width:1200px; margin:0 auto 48px; padding:0 20px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin:24px 0; }
.card { background:var(--card); border-radius:16px; padding:18px; box-shadow:0 10px 30px rgba(0,0,0,.25); }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:700; }
.ok { background:#064e3b; color:var(--ok); }
.bad { background:#7f1d1d; color:var(--bad); }
.run { background:#78350f; color:var(--run); }
.lesson { margin:18px 0; }
.images { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; margin-top:12px; }
.images img { width:100%; border-radius:12px; background:#0b1220; }
pre { overflow:auto; background:#0b1220; padding:12px; border-radius:10px; font-size:12px; }
a { color:var(--accent); }
.topnav { display:flex; gap:18px; justify-content:center; flex-wrap:wrap; margin-top:14px; }
</style>
</head>
<body>
<header>
  <h1>Deep Learning Lab 验证看板</h1>
  <p class="sub">先读原理，再单独跑实验，这里汇总指标与图像</p>
  <div class="topnav">
    <a href="/learn/">学习导读</a>
    <a href="/learn/01">感知机</a>
    <a href="/learn/02">MLP</a>
    <a href="/learn/03">反向传播</a>
    <a href="/learn/04">激活函数</a>
    <a href="/learn/05">学习率</a>
    <a href="/learn/06">参数初始化</a>
    <a href="/learn/07">归一化</a>
    <a href="/learn/08">正则化</a>
    <a href="/learn/09">MNIST</a>
    <a href="/docs/architecture">MNIST 架构图</a>
  </div>
</header>
<div class="wrap">
  <div class="grid">
    <div class="card"><div>总体状态</div><h2><span class="badge {{ 'ok' if summary.passed else 'bad' }}">{{ 'PASS' if summary.passed else 'FAIL' }}</span></h2></div>
    <div class="card"><div>实验数量</div><h2>{{ summary.lessons|length }}</h2></div>
    <div class="card"><div>开始时间</div><h2 style="font-size:1rem">{{ summary.started_at or '-' }}</h2></div>
    <div class="card"><div>结束时间</div><h2 style="font-size:1rem">{{ summary.finished_at or '进行中' }}</h2></div>
  </div>
  <section class="card">
    <h2>单独运行</h2>
    <pre>python scripts/run_lesson.py --list
python scripts/run_lesson.py 01
python scripts/run_lesson.py 08
python scripts/run_lesson.py 09 --epochs 3</pre>
  </section>
  {% for lesson in summary.lessons %}
  {% set images = (lesson.metrics or {}).get('images') or lesson.get('images') or [] %}
  <section class="card lesson">
    <h2>{{ lesson.name }} <span class="badge {{ 'ok' if lesson.status=='passed' else ('run' if lesson.status=='running' else 'bad') }}">{{ lesson.status or ('passed' if lesson.passed else 'unknown') }}</span></h2>
    <p class="sub">耗时 {{ lesson.elapsed_sec or '-' }} s · <a href="/learn/{{ lesson.name.split('_')[0] }}">阅读原理</a></p>
    {% if lesson.error %}<pre>{{ lesson.error }}</pre>{% endif %}
    {% if lesson.metrics %}
      <pre>{{ lesson.metrics | tojson(indent=2) }}</pre>
    {% endif %}
    {% if images %}
      <div class="images">
        {% for img in images %}
          <a href="/file/{{ img }}"><img src="/file/{{ img }}" alt="{{ img }}"></a>
        {% endfor %}
      </div>
    {% endif %}
  </section>
  {% endfor %}
  <p><a href="/api/summary">JSON API</a></p>
</div>
</body>
</html>
"""

DOC_FILES = {
    "01": "01_perceptron.html",
    "02": "02_mlp.html",
    "03": "03_backprop.html",
    "03-visual": "03_backprop.html",
    "04": "04_activations.html",
    "05": "05_learning_rate.html",
    "06": "06_weight_init.html",
    "07": "07_normalization.html",
    "08": "08_regularization.html",
    "09": "09_mnist.html",
}


def load_summary() -> dict:
    path = ARTIFACTS_DIR / "summary.json"
    if not path.exists():
        return {"passed": False, "lessons": [], "started_at": None, "finished_at": None}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/")
def index():
    return render_template_string(PAGE, summary=load_summary())


@app.get("/api/summary")
def api_summary():
    return load_summary()


@app.get("/learn.css")
@app.get("/learn/learn.css")
def learn_css():
    return send_from_directory(DOCS, "learn.css")


@app.get("/learn/")
@app.get("/learn")
def learn_index():
    return send_from_directory(DOCS, "index.html")


@app.get("/learn/<key>")
def learn_page(key: str):
    # Support both short keys (/learn/01) and relative filenames
    # (/learn/01_perceptron.html) so docs work via Flask and file://.
    if key.endswith((".html", ".css")):
        name = key
    else:
        name = DOC_FILES.get(key)
        if name is None:
            try:
                name = find_lesson(key)["doc"]
            except KeyError:
                abort(404)
    path = (DOCS / name).resolve()
    if DOCS.resolve() not in path.parents and path != DOCS.resolve():
        abort(403)
    if not path.exists() or not path.is_file():
        abort(404)
    return send_from_directory(DOCS, name)


@app.get("/file/<path:rel>")
def artifact_file(rel: str):
    target = (PROJECT_ROOT / rel).resolve()
    if PROJECT_ROOT not in target.parents and target != PROJECT_ROOT:
        abort(403)
    if not target.exists():
        abort(404)
    return send_from_directory(target.parent, target.name)


@app.get("/docs/architecture")
def architecture():
    if not ARCH_DOC.exists():
        abort(404)
    return Response(ARCH_DOC.read_text(encoding="utf-8"), mimetype="text/html")


def main():
    app.run(host="0.0.0.0", port=8080, debug=False)


if __name__ == "__main__":
    main()
