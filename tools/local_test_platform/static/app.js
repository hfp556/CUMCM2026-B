"use strict";

const $ = (selector) => document.querySelector(selector);
const elements = {
  form: $("#scenarioForm"), seed: $("#seed"), count: $("#sourceCount"),
  error: $("#errorMode"), failEvery: $("#failEvery"), forceFail: $("#forceClearFail"),
  version: $("#version"), reuse: $("#reuseScenario"), run: $("#runButton"),
  prepare: $("#prepareButton"), stop: $("#stopButton"), canvas: $("#arena"),
  truth: $("#showTruth"), body: $("#transactionBody"), packet: $("#packetDetail"),
  output: $("#programOutput"), toast: $("#toast"), importFile: $("#importFile"),
};

function selectedRunMode() {
  return document.querySelector('input[name="run_mode"]:checked').value;
}

let latestState = null;
let lastGeneration = -1;
let lastOutputCount = -1;
let polling = false;
let toastTimer = null;

$("#apiAddress").textContent = window.location.host;

const statusLabels = {
  idle: "空闲", starting: "正在启动", running: "运行中", stopping: "正在停止",
  stopped: "已停止", completed: "已完成", failed: "运行失败",
};

function toast(message, isError = false) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.className = `toast show${isError ? " error" : ""}`;
  toastTimer = setTimeout(() => { elements.toast.className = "toast"; }, 2600);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
  if (!response.ok) throw new Error(data.error || `请求失败：HTTP ${response.status}`);
  return data;
}

function scenarioPayload() {
  const interval = elements.failEvery.value.trim();
  return {
    seed: Number(elements.seed.value),
    source_count: Number(elements.count.value),
    error_mode: elements.error.value,
    fail_transport_every: interval ? Number(interval) : null,
    force_first_clear_fail: elements.forceFail.checked,
  };
}

function number(value, digits = 1) {
  return Number(value || 0).toLocaleString("zh-CN", { maximumFractionDigits: digits });
}

function resultOf(transaction) {
  const response = transaction.response;
  if (!response) return ["连接中断", "result-fail"];
  if (response.accepted === false) return [response.error || "未执行", "result-fail"];
  const value = response.measure_result || response.clear_result || response.exit_reason || "accepted";
  if (value === "success") return ["清除成功", "result-success"];
  if (value === "direction") return [`方向 ${number(response.svd_deg, 2)}°`, "result-direction"];
  if (value === "near") return ["距离过近", "result-direction"];
  if (value === "no_signal") return ["无信号", ""];
  if (value === "no_target_in_range") return ["未清除", "result-fail"];
  if (value === "user_exit") return ["主动退出", ""];
  return [value === "accepted" ? "已接受" : value, ""];
}

function renderMetrics(state) {
  const s = state.summary;
  const session = state.session;
  $("#metricTime").textContent = `${number(s.total_virtual_time_s)} s`;
  $("#metricMinutes").textContent = `${number(s.total_virtual_time_s / 60)} min`;
  $("#metricCleared").textContent = `${s.discovered_count} / ${s.cleared_count}`;
  const missing = s.uncleared_channels.length;
  $("#metricMissed").textContent = missing ? `${missing} 个信号源尚未清除` : (s.source_count ? "当前全部清除" : "等待进入");
  $("#metricMove").textContent = `${number(s.total_move_m, 0)} m`;
  $("#metricActions").textContent = `${state.actions.length} 次动作 · ${s.clear_failure_count} 次清除失败`;
  $("#metricPosition").textContent = `${number(session.position[0], 0)}, ${number(session.position[1], 0)}`;
  $("#metricChannel").textContent = `测向频道 ${session.current_channel}`;
}

function renderStatus(state) {
  const running = ["starting", "running", "stopping"].includes(state.run.status);
  const progress = state.run.mode === "batch" && running
    ? ` · ${state.batch.completed}/${state.batch.total}` : "";
  $("#runStatus").textContent = (statusLabels[state.run.status] || state.run.status) + progress;
  elements.run.disabled = running;
  elements.prepare.disabled = running;
  elements.stop.disabled = !running;
  const batch = selectedRunMode() === "batch";
  elements.reuse.disabled = batch;
  const runLabel = batch
    ? "运行 12 个不同种子"
    : (elements.reuse.checked ? "沿用场景并运行" : "创建场景并运行");
  elements.run.querySelector("span:last-child").textContent = runLabel;
}

function renderBatch(state) {
  const panel = $("#batchPanel");
  const show = state.run.mode === "batch" || selectedRunMode() === "batch";
  panel.hidden = !show;
  if (!show) return;
  const batch = state.batch;
  const average = batch.average_virtual_time_s;
  const version = state.run.version || elements.version.value;
  $("#batchTitle").textContent = batch.total
    ? `${version} · 已完成 ${batch.completed}/${batch.total}${batch.current_seed ? ` · 当前 seed ${batch.current_seed}` : ""}`
    : `${version} · 等待开始`;
  $("#batchAverage").textContent = average == null ? "—" : `${number(average)} s`;
  $("#batchSuccess").textContent = `${batch.successful} / ${batch.total || 12}`;
  $("#batchActions").textContent = batch.average_action_count == null ? "—" : number(batch.average_action_count, 1);
  $("#batchProgress").style.width = `${batch.total ? batch.completed / batch.total * 100 : 0}%`;

  const resultBySeed = new Map(batch.results.map((item) => [item.seed, item]));
  const seeds = batch.seeds.length
    ? batch.seeds
    : Array.from({ length: 12 }, (_, index) => Number(elements.seed.value) + index);
  $("#batchRuns").innerHTML = seeds.map((seed, index) => {
    const result = resultBySeed.get(seed);
    const current = batch.current_seed === seed;
    const css = result ? (result.successful ? "success" : "failed") : (current ? "current" : "");
    const label = result ? `${number(result.total_virtual_time_s)}s` : (current ? "运行中" : `${index + 1}/12`);
    return `<div class="batch-run ${css}"><span>${seed}</span><strong>${label}</strong></div>`;
  }).join("");
}

function renderTransactions(state) {
  const rows = state.transactions.slice(-100).reverse();
  $("#transactionCount").textContent = `${state.transactions.length} 条`;
  if (!rows.length) {
    elements.body.innerHTML = '<tr><td colspan="5" class="empty">等待程序连接</td></tr>';
    return;
  }
  elements.body.innerHTML = rows.map((item) => {
    const [label, className] = resultOf(item);
    const channel = item.payload.channel ?? "—";
    const virtualTime = item.response?.accepted ? `${number(item.response.virtual_time_s)} s` : "—";
    return `<tr data-sequence="${item.sequence}"><td>${item.sequence}</td><td>${item.path}</td>` +
      `<td>${channel}</td><td class="${className}">${label}</td><td>${virtualTime}</td></tr>`;
  }).join("");
}

function renderOutput(state) {
  if (state.output.length === lastOutputCount) return;
  const stickToBottom = elements.output.scrollTop + elements.output.clientHeight >= elements.output.scrollHeight - 28;
  lastOutputCount = state.output.length;
  if (!state.output.length) {
    elements.output.innerHTML = '<div class="output-placeholder">运行程序后，这里会实时显示输出。</div>';
    return;
  }
  elements.output.innerHTML = state.output.map((line) =>
    `<div class="output-line ${line.stream === "platform" ? "platform" : ""}">` +
    `<span class="output-time">${line.wall_time}</span><span class="output-text"></span></div>`
  ).join("");
  elements.output.querySelectorAll(".output-text").forEach((node, index) => {
    node.textContent = state.output[index].text;
  });
  if (stickToBottom || state.output.length < 8) elements.output.scrollTop = elements.output.scrollHeight;
}

function drawArena(state) {
  const canvas = elements.canvas;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.floor(rect.width * dpr);
  const height = Math.floor(rect.height * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width; canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const scale = Math.min(rect.width, rect.height) * .43 / 1800;
  const point = ([x, y]) => [cx + x * scale, cy - y * scale];

  ctx.strokeStyle = "rgba(86, 128, 154, .11)";
  ctx.lineWidth = 1;
  for (let radius = 300; radius <= 1800; radius += 300) {
    ctx.beginPath(); ctx.arc(cx, cy, radius * scale, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.beginPath(); ctx.moveTo(cx - 1800 * scale, cy); ctx.lineTo(cx + 1800 * scale, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy - 1800 * scale); ctx.lineTo(cx, cy + 1800 * scale); ctx.stroke();
  ctx.strokeStyle = "rgba(85, 219, 228, .36)";
  ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.arc(cx, cy, 1800 * scale, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = "rgba(145, 163, 178, .55)";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText("1800 m", cx + 1800 * scale - 48, cy - 8);

  const positions = [[0, 0], ...state.actions.map((action) => action.position)];
  if (positions.length > 1) {
    ctx.strokeStyle = "rgba(150, 177, 194, .55)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    positions.forEach((position, index) => {
      const [x, y] = point(position);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  state.actions.slice(-500).forEach((action) => {
    const [x, y] = point(action.position);
    const isClear = action.path === "/clear";
    const success = action.result === "success";
    ctx.beginPath();
    ctx.arc(x, y, isClear ? 4.5 : 2.5, 0, Math.PI * 2);
    ctx.fillStyle = isClear ? (success ? "#6fe5a5" : "#ff6f70") : "rgba(85, 219, 228, .85)";
    ctx.fill();
    if (isClear) {
      ctx.strokeStyle = ctx.fillStyle; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.stroke();
    }
  });

  if (elements.truth.checked) {
    state.scenario.sources.forEach((source) => {
      const [x, y] = point(source.position);
      const cleared = state.summary.cleared_channels.includes(source.channel);
      ctx.beginPath(); ctx.arc(x, y, 5.5, 0, Math.PI * 2);
      ctx.fillStyle = cleared ? "rgba(111, 229, 165, .28)" : "#ffbc66";
      ctx.fill();
      ctx.strokeStyle = cleared ? "#6fe5a5" : "rgba(255, 188, 102, .5)";
      ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = cleared ? "#7ea894" : "#ffd8a6";
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(String(source.channel), x + 9, y - 7);
    });
  }

  const [robotX, robotY] = point(state.session.position);
  ctx.fillStyle = "#e9f1f6";
  ctx.beginPath(); ctx.arc(robotX, robotY, 5, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "rgba(233, 241, 246, .35)";
  ctx.beginPath(); ctx.arc(robotX, robotY, 10, 0, Math.PI * 2); ctx.stroke();

  const last = state.actions[state.actions.length - 1];
  $("#lastAction").textContent = last
    ? `${last.path} · CH ${last.channel} · (${number(last.position[0], 0)}, ${number(last.position[1], 0)}) · ${last.result}`
    : "尚无动作";
}

function syncControls(state) {
  if (state.generation === lastGeneration) return;
  lastGeneration = state.generation;
  elements.seed.value = state.config.seed;
  elements.count.value = state.config.source_count;
  elements.error.value = state.config.error_mode;
  elements.failEvery.value = state.config.fail_transport_every || "";
  elements.forceFail.checked = Boolean(state.config.force_first_clear_fail);
}

function render(state) {
  latestState = state;
  syncControls(state);
  renderStatus(state);
  renderMetrics(state);
  renderBatch(state);
  renderTransactions(state);
  renderOutput(state);
  drawArena(state);
}

async function poll() {
  if (polling) return;
  polling = true;
  try {
    const state = await request("/api/state");
    $("#connectionDot").className = "status-dot online";
    $("#connectionLabel").textContent = "本地模拟器已连接";
    render(state);
  } catch (error) {
    $("#connectionDot").className = "status-dot offline";
    $("#connectionLabel").textContent = "连接已断开";
  } finally {
    polling = false;
  }
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const mode = selectedRunMode();
    const payload = {
      ...scenarioPayload(),
      version: elements.version.value,
      run_mode: mode,
      reuse_scenario: mode === "single" && elements.reuse.checked,
    };
    render(await request("/api/run", { method: "POST", body: JSON.stringify(payload) }));
    toast(mode === "batch" ? "已开始运行 12 个不同种子" : `已启动 ${elements.version.options[elements.version.selectedIndex].text}`);
  } catch (error) { toast(error.message, true); }
});

elements.prepare.addEventListener("click", async () => {
  try {
    render(await request("/api/scenario", { method: "POST", body: JSON.stringify(scenarioPayload()) }));
    elements.reuse.checked = true;
    renderStatus(latestState);
    toast("新场景已准备，可以手动调试或直接运行");
  } catch (error) { toast(error.message, true); }
});

elements.stop.addEventListener("click", async () => {
  try { render(await request("/api/stop", { method: "POST", body: "{}" })); }
  catch (error) { toast(error.message, true); }
});

$("#randomSeed").addEventListener("click", () => {
  elements.seed.value = Math.floor(10000000 + Math.random() * 89999999);
  elements.reuse.checked = false;
  if (latestState) renderStatus(latestState);
});
elements.reuse.addEventListener("change", () => latestState && renderStatus(latestState));
document.querySelectorAll('input[name="run_mode"]').forEach((input) => {
  input.addEventListener("change", () => {
    if (latestState) {
      renderStatus(latestState);
      renderBatch(latestState);
    }
  });
});
elements.truth.addEventListener("change", () => latestState && drawArena(latestState));
window.addEventListener("resize", () => latestState && drawArena(latestState));

elements.body.addEventListener("click", (event) => {
  const row = event.target.closest("tr[data-sequence]");
  if (!row || !latestState) return;
  const item = latestState.transactions.find((entry) => entry.sequence === Number(row.dataset.sequence));
  if (item) elements.packet.textContent = JSON.stringify(item, null, 2);
});

$("#copyOutput").addEventListener("click", async () => {
  if (!latestState) return;
  const output = latestState.output.map((line) => line.text).join("\n");
  try { await navigator.clipboard.writeText(output); toast("程序输出已复制"); }
  catch (_error) { toast("浏览器未允许复制，请手动选择日志", true); }
});

$("#exportButton").addEventListener("click", () => { window.location.href = "/api/export"; });
$("#importButton").addEventListener("click", () => elements.importFile.click());
elements.importFile.addEventListener("change", async () => {
  const file = elements.importFile.files[0];
  if (!file) return;
  try {
    const exported = JSON.parse(await file.text());
    if (!exported.scenario?.sources) throw new Error("文件中没有可复现的场景");
    const payload = {
      ...exported.config,
      seed: exported.scenario.seed,
      sources: exported.scenario.sources,
    };
    render(await request("/api/scenario", { method: "POST", body: JSON.stringify(payload) }));
    elements.reuse.checked = true;
    renderStatus(latestState);
    toast("场景已导入；勾选“沿用当前信号源位置”后运行即可复现");
  } catch (error) { toast(error.message, true); }
  elements.importFile.value = "";
});

document.querySelector(".manual-actions").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-path]");
  if (!button) return;
  const path = button.dataset.path;
  const payload = {
    arena_id: "default",
    robot_id: $("#robotId").value.trim(),
    request_id: `manual-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
  };
  if (["/measure", "/clear"].includes(path)) {
    payload.position = { x: Number($("#manualX").value), y: Number($("#manualY").value) };
    payload.channel = Number($("#manualChannel").value);
  }
  try {
    const response = await request(path, { method: "POST", body: JSON.stringify(payload) });
    elements.packet.textContent = JSON.stringify({ path, payload, response }, null, 2);
    toast(`${path} 已执行`);
    await poll();
  } catch (error) { toast(error.message, true); }
});

setInterval(poll, 650);
poll();
