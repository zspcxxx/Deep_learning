// Minimal canvas line chart: lineChart(canvas, {series: [{label, color, points: [[x, y], ...], dots, dashed}], xLabel}).
(function () {
  function niceTicks(min, max, count) {
    if (!isFinite(min) || !isFinite(max)) return [0, 1];
    if (min === max) { min -= 1; max += 1; }
    const raw = (max - min) / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
    return ticks;
  }

  function fmt(v) {
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
    if (a >= 1000) return (v / 1000).toFixed(a >= 10000 ? 0 : 1) + "k";
    return String(+v.toFixed(4));
  }

  function lineChart(canvas, opts) {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 400;
    const cssH = canvas.height / (canvas._dpr || 1);
    canvas._dpr = dpr;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const series = (opts.series || []).filter((s) => s.points && s.points.length);
    const pad = { l: 48, r: 12, t: 22, b: 30 };
    const w = cssW - pad.l - pad.r;
    const h = cssH - pad.t - pad.b;
    ctx.font = "11px sans-serif";
    if (!series.length) {
      ctx.fillStyle = "#9ca3af";
      ctx.textAlign = "center";
      ctx.fillText(opts.empty || "暂无数据", cssW / 2, cssH / 2);
      return;
    }
    const xs = series.flatMap((s) => s.points.map((p) => p[0]));
    const ys = series.flatMap((s) => s.points.map((p) => p[1])).filter((v) => v !== null && isFinite(v));
    let xMin = Math.min(...xs), xMax = Math.max(...xs);
    if (xMin === xMax) { xMin -= 1; xMax += 1; }
    let yMin = opts.yMin !== undefined ? opts.yMin : Math.min(...ys);
    let yMax = Math.max(...ys);
    const span = yMax - yMin || Math.abs(yMax) || 1;
    if (opts.yMin === undefined) yMin -= span * 0.05;
    yMax += span * 0.08;
    const X = (v) => pad.l + ((v - xMin) / (xMax - xMin)) * w;
    const Y = (v) => pad.t + h - ((v - yMin) / (yMax - yMin)) * h;

    ctx.strokeStyle = "#eef2f7";
    ctx.fillStyle = "#6b7280";
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of niceTicks(yMin, yMax, 5)) {
      if (t < yMin || t > yMax) continue;
      ctx.beginPath(); ctx.moveTo(pad.l, Y(t)); ctx.lineTo(pad.l + w, Y(t)); ctx.stroke();
      ctx.fillText(fmt(t), pad.l - 6, Y(t));
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of niceTicks(xMin, xMax, 6)) {
      if (t < xMin || t > xMax) continue;
      ctx.fillText(fmt(t), X(t), pad.t + h + 6);
    }
    ctx.strokeStyle = "#cbd5e1";
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + h); ctx.lineTo(pad.l + w, pad.t + h); ctx.stroke();
    if (opts.xLabel) {
      ctx.textAlign = "right";
      ctx.fillText(opts.xLabel, pad.l + w, pad.t + h + 18);
    }

    let lx = pad.l + 6;
    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.fillStyle = s.color;
      ctx.lineWidth = s.width || 1.6;
      ctx.setLineDash(s.dashed ? [5, 4] : []);
      ctx.beginPath();
      let started = false;
      for (const [x, y] of s.points) {
        if (y === null || !isFinite(y)) continue;
        if (!started) { ctx.moveTo(X(x), Y(y)); started = true; } else ctx.lineTo(X(x), Y(y));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (s.dots) {
        for (const [x, y] of s.points) {
          if (y === null || !isFinite(y)) continue;
          ctx.beginPath(); ctx.arc(X(x), Y(y), 3, 0, Math.PI * 2); ctx.fill();
        }
      }
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillRect(lx, 8, 12, 3);
      ctx.fillStyle = "#374151";
      ctx.fillText(s.label, lx + 16, 10);
      lx += 24 + ctx.measureText(s.label).width;
    }
  }

  window.lineChart = lineChart;
})();
