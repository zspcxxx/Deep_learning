// Lesson dropdown for the top nav of every doc page. Add new lessons here only.
(function () {
  const GROUPS = [
    ["基础", [
      ["01_perceptron.html", "01", "感知机", "线性边界，XOR 学不会"],
      ["02_mlp.html", "02", "多层感知机 MLP", "隐藏层带来非线性"],
      ["03_backprop.html", "03", "反向传播", "链式法则与梯度检查"],
      ["04_activations.html", "04", "激活函数", "非线性与梯度消失"],
    ]],
    ["训练技巧", [
      ["05_learning_rate.html", "05", "学习率", "步长决定能否收敛"],
      ["06_weight_init.html", "06", "参数初始化", "全零 / 过小 / 过大 / He"],
      ["07_normalization.html", "07", "归一化", "输入标准化与 BatchNorm"],
      ["08_regularization.html", "08", "正则化", "L1 / L2 / Dropout / 早停"],
      ["09_full_pipeline.html", "09", "综合实战：MNIST", "11 项技术识别手写数字"],
    ]],
    ["网络结构", [
      ["10_cnn.html", "10", "卷积神经网络 CNN", "检测图片里的小猫"],
      ["11_residual.html", "11", "残差连接", "让 50 层网络也能训练"],
      ["12_rnn.html", "12", "RNN / LSTM / GRU", "序列与长期记忆"],
      ["13_seq2seq.html", "13", "编码器–解码器", "Seq2Seq 倒序任务"],
      ["14_attention.html", "14", "注意力机制", "Q / K / V 按键查值"],
    ]],
    ["综合", [
      ["15_transformer.html", "15", "综合串联：从零组装 Transformer", "前 14 课的零件拼起来，再一个个拿掉"],
    ]],
  ];

  const nav = document.querySelector("nav.top");
  const slot = nav && nav.querySelector("[data-lesson-menu]");
  if (!slot) return;

  const here = location.pathname.split("/").pop() || "index.html";
  const items = GROUPS.flatMap(([, list]) => list);
  const isHere = ([file, num]) => file === here || num === here;
  const current = items.find(isHere);

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text) node.textContent = text;
    return node;
  };

  const button = el("button", "menu-button");
  button.type = "button";
  button.setAttribute("aria-haspopup", "true");
  button.setAttribute("aria-expanded", "false");
  button.append(
    el("span", "menu-label", current ? `${current[1]} ${current[2]}` : "选择课程"),
    el("span", "menu-caret", "▾"),
  );

  const panel = el("div", "menu-panel");
  panel.hidden = true;
  for (const [group, list] of GROUPS) {
    const section = el("div", group === "综合" ? "menu-group wide" : "menu-group");
    section.append(el("div", "menu-group-title", group));
    for (const entry of list) {
      const [file, num, title, sub] = entry;
      const link = el("a", "menu-item");
      link.href = file;
      if (entry === current) {
        link.classList.add("current");
        link.setAttribute("aria-current", "page");
      }
      link.append(el("span", "menu-num", num), el("span", "menu-title", title), el("span", "menu-sub", sub));
      section.append(link);
    }
    panel.append(section);
  }

  const setOpen = (open) => {
    panel.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
    if (open) (panel.querySelector(".current") || panel.querySelector("a")).focus();
  };
  button.addEventListener("click", () => setOpen(panel.hidden));
  document.addEventListener("click", (e) => { if (!slot.contains(e.target)) setOpen(false); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !panel.hidden) { setOpen(false); button.focus(); }
  });

  slot.append(button, panel);
})();
