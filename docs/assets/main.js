// Mobile nav
const navToggle = document.getElementById("navToggle");
const nav = document.getElementById("nav");
if (navToggle && nav) navToggle.addEventListener("click", () => nav.classList.toggle("open"));

// Footer year
const yearEl = document.getElementById("year");
if (yearEl) yearEl.textContent = new Date().getFullYear();

// Toggle task details
const btn = document.getElementById("toggleTaskDetails");
const details = document.getElementById("taskDetails");
if (btn && details) btn.addEventListener("click", () => details.classList.toggle("open"));

// Render task summary (Table 1)
const taskGrid = document.getElementById("taskGrid");
const tasks = [
  {
    group: "Mental Rotation",
    desc: "Mentally representing and rotating objects while maintaining their features;",
    items: [
      {
        name: "2D Rotation",
        obj: "Identify correct 2D rotation.",
        neg: "Non-centrally symmetric patterns; mirroring; view mirroring.",
        diff: "Internal pattern rotation.",
      },
      {
        name: "3D Rotation",
        obj: "Identify correct 3D rotation.",
        neg: "Wrong view substitution; cube removal.",
        diff: "Larger assemblies.",
      },
      {
        name: "Three-View Projection",
        obj: "Select left view from projections (engineering parts).",
        neg: "View flipping; line deletion.",
        diff: "Real engineering parts (DeepCAD).",
      },
    ],
  },
  {
    group: "Mental Folding",
    desc: "Mentally folding two-dimensional patterns into three-dimensional objects or vice versa;",
    items: [
      {
        name: "Paper Folding",
        obj: "Predict unfolded hole pattern.",
        neg: "Hole mirroring, addition, deletion, or relocation.",
        diff: "More folds; larger grid; more holes.",
      },
      {
        name: "Cube Unfolding",
        obj: "Select correct 2D net from view.",
        neg: "Asymmetric/dot patterns on faces; swapping face colors.",
        diff: "Rotating internal patterns.",
      },
      {
        name: "Cube Reconstruction",
        obj: "Select 3D view from net; find opposite face.",
        neg: "Mirroring the correct 3D view.",
        diff: "Follows Cube Unfolding.",
      },
    ],
  },
  {
    group: "Visual Penetration",
    desc: "Imagining the internal structure of objects based on external features;",
    items: [
      {
        name: "Cross-Section",
        obj: "Identify cross-section of solid.",
        neg: "Altered geometric proportions.",
        diff: "3-solid composites; oblique slicing.",
      },
      {
        name: "Cube Counting",
        obj: "Infer total cube count from views.",
        neg: "Options from min/max math bounds.",
        diff: "2 to 3 views; larger assemblies.",
      },
      {
        name: "Cube Assembly",
        obj: "Find complementary part of split stack.",
        neg: "Add/remove cubes from correct part.",
        diff: "Larger stacks; 3-part splits.",
      },
    ],
  },
  {
    group: "Mental Animation",
    desc: "Mentally visualizing the motion and movement of components within a system.",
    items: [
      {
        name: "Arrow Moving",
        obj: "Predict final state.",
        neg: "Incorrect endpoint from same start.",
        diff: "Multiple arrows; movement sequence interaction rules.",
      },
      {
        name: "Block Moving",
        obj: "Predict final state with gravity.",
        neg: "Incorrect final states.",
        diff: "Higher complexity; longer sequences.",
      },
      {
        name: "Mechanical System",
        obj: "Understand motion propagation.",
        neg: "Incorrect motion outcomes.",
        diff: "More system modules.",
      },
    ],
  },
];
if (taskGrid) {
  taskGrid.innerHTML = tasks.map((t) => `
    <div class="card">
      <h3>${t.group}</h3>
      <p>${t.desc}</p>
      ${t.items.map((x) => `
        <div class="task-item">
          <h4>${x.name}</h4>
          <ul class="bullets compact">
            <li><b>Objective:</b> ${x.obj}</li>
            <li><b>Negative samples:</b> ${x.neg}</li>
            <li><b>Difficulty:</b> ${x.diff}</li>
          </ul>
        </div>
      `).join("")}
    </div>
  `).join("");
}

/* -----------------------------
   Examples carousel
-------------------------------- */
const exTrack = document.getElementById("exTrack");
const exPrev = document.getElementById("exPrev");
const exNext = document.getElementById("exNext");
const examples = [
  {
    title: "Mental Rotation • 2D Rotation",
    desc: "Example: identify the correct 2D rotation among options.",
    img: "./assets/images/examples/2D_rotation.png",
  },
  {
    title: "Mental Rotation • 3D Rotation",
    desc: "Example: rotate a 3D object and choose the correct view among image options.",
    img: "./assets/images/examples/3D_rotation.png",
  },
  {
    title: "Mental Rotation • Three-View Projection",
    desc: "Example: select the correct left view from projections.",
    img: "./assets/images/examples/3view_projection.png",
  },
  {
    title: "Mental Folding • Paper Folding",
    desc: "Example: infer the unfolded hole pattern after folds.",
    img: "./assets/images/examples/paper_folding.png",
  },
  {
    title: "Mental Folding • Cube Unfolding",
    desc: "Example: select the correct 2D net from a 3D cube view.",
    img: "./assets/images/examples/cube_unfolding.png",
  },
  {
    title: "Mental Folding • Cube Reconstruction",
    desc: "Example: select the correct 3D cube from a net.",
    img: "./assets/images/examples/cube_reconstruction.png",
  },
  {
    title: "Visual Penetration • Cross Section",
    desc: "Example: predict cross-section shape under a cutting plane.",
    img: "./assets/images/examples/cross_section.png",
  },
  {
    title: "Visual Penetration • Cube Counting",
    desc: "Example: infer total cube count from multiple views.",
    img: "./assets/images/examples/cube_counting.png",
  },
  {
    title: "Visual Penetration • Cube Assembly",
    desc: "Example: find the complementary part of a split stack.",
    img: "./assets/images/examples/cube_assembly.png",
  },
  {
    title: "Mental Animation • Arrow Moving",
    desc: "Example: predict final state from movement rules.",
    img: "./assets/images/examples/arrow_moving.png",
  },
  {
    title: "Mental Animation • Block Moving",
    desc: "Example: predict the final state with gravity.",
    img: "./assets/images/examples/block_moving.png",
  },
  {
    title: "Mental Animation • Mechanical System",
    desc: "Example: propagate motion through linked components.",
    img: "./assets/images/examples/mechanical_system.png",
  },
];

function renderExamples() {
  if (!exTrack) return;
  exTrack.innerHTML = examples.map((e) => `
    <div class="slide" style="padding:0;gap:0;align-items:start;min-height:auto">
      <div class="card" style="padding:10px;margin:0">
        <h3 style="margin:0 0 4px 0">${e.title}</h3>
        <p style="margin:0">${e.desc}</p>
      </div>
      <button class="img-btn" data-img="${e.img}" data-cap="${e.title}" style="margin:0;padding:0;display:block;line-height:0">
        <img src="${e.img}" alt="${e.title}" style="width:100%;margin:0;display:block;max-height:none;height:auto" />
      </button>
    </div>
  `).join("");
}

let exIndex = 0;
function goToExample(i) {
  if (!exTrack) return;
  exIndex = Math.max(0, Math.min(i, examples.length - 1));
  exTrack.scrollTo({ left: exIndex * exTrack.clientWidth, behavior: "smooth" });
}
if (exPrev) exPrev.addEventListener("click", () => goToExample(exIndex - 1));
if (exNext) exNext.addEventListener("click", () => goToExample(exIndex + 1));
window.addEventListener("resize", () => goToExample(exIndex));
renderExamples();

/* -----------------------------
   Image modal (click to enlarge)
-------------------------------- */
const imgModal = document.getElementById("imgModal");
const modalImg = document.getElementById("modalImg");
const modalCap = document.getElementById("modalCap");
const modalClose = document.getElementById("modalClose");
const modalX = document.getElementById("modalX");

function openModal(src, cap) {
  if (!imgModal || !modalImg || !modalCap) return;
  modalImg.src = src;
  modalCap.textContent = cap || "";
  imgModal.classList.add("open");
  imgModal.setAttribute("aria-hidden", "false");
}
function closeModal() {
  if (!imgModal) return;
  imgModal.classList.remove("open");
  imgModal.setAttribute("aria-hidden", "true");
}
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".img-btn");
  if (!btn) return;
  const src = btn.getAttribute("data-img");
  const cap = btn.getAttribute("data-cap");
  if (src) openModal(src, cap);
});
if (modalClose) modalClose.addEventListener("click", closeModal);
if (modalX) modalX.addEventListener("click", closeModal);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

/* -----------------------------
   Leaderboard (Paper Table 2)
-------------------------------- */
// Fields:
// - overall: single reported overall
// - direct / cot: both reported
const leaderboard = [
  { model: "Gemini-2.5-pro", source: "Closed", overall: 44.66 },
  { model: "o1", source: "Closed", overall: 41.36 },
  { model: "Doubao-1-5-vision-pro", source: "Closed", direct: 37.54, cot: 33.31 },
  { model: "Gemini-2.5-flash", source: "Closed", overall: 36.86 },
  { model: "Qwen-VL-max", source: "Closed", direct: 36.10, cot: 32.03 },
  { model: "Qwen2.5-VL-72B-Instruct", source: "Open", direct: 35.00, cot: 33.31 },
  { model: "Qwen2.5-VL-32B-Instruct", source: "Open", direct: 33.90, cot: 32.12 },
  { model: "Claude-3.7-sonnet", source: "Closed", overall: 33.90 },
  { model: "Claude-3.5-sonnet", source: "Closed", direct: 26.86, cot: 32.54 },
  { model: "Kimi-VL-A3B-Instruct(16B)", source: "Open", direct: 32.37, cot: 23.90 },
  { model: "InternVL3-78B", source: "Open", direct: 31.69, cot: 25.83 },
  { model: "QvQ-72B-preview", source: "Open", direct: 31.25, cot: 30.93 },
  { model: "Qwen2.5-Omni-7B", source: "Open", direct: 31.44, cot: 27.29 },
  { model: "Deepseek-VL2(27B)", source: "Open", direct: 31.15, cot: 25.97 },
  { model: "InternVL3-38B", source: "Open", direct: 31.02, cot: 25.42 },
  { model: "Qwen2.5-VL-7B-Instruct", source: "Open", direct: 30.76, cot: 27.97 },
  { model: "GPT-4o", source: "Closed", direct: 30.76, cot: 31.10 },
  { model: "InternVL3-8B", source: "Open", direct: 30.25, cot: 30.08 },
  { model: "Kimi-VL-A3B-thinking(16B)", source: "Open", direct: 30.08, cot: 21.61 },
  { model: "Qwen2.5-VL-3B-Instruct", source: "Open", direct: 30.17, cot: 26.10 },
  { model: "SAIL-VL-1.5-2B", source: "Open", direct: 29.32, cot: 24.15 },
  { model: "Deepseek-VL2-tiny(3B)", source: "Open", direct: 29.58, cot: 21.36 },
  { model: "SAIL-VL-1.6-8B", source: "Open", direct: 29.15, cot: 25.00 },
  { model: "InternVL3-2B", source: "Open", overall: 26.19 },
  { model: "Deepseek-VL2-small(16B)", source: "Open", direct: 27.63, cot: 22.45 },
  { model: "Llama-4-Maverick-17B-128E-Instruct", source: "Open", direct: 32.03, cot: 26.86 },
];

// Helpers
const fmt = (x) => (typeof x === "number" ? x.toFixed(2) : "-");
const getVal = (m, metric) => {
  if (metric === "direct") return (typeof m.direct === "number" ? m.direct : (typeof m.overall === "number" ? m.overall : null));
  if (metric === "cot") return (typeof m.cot === "number" ? m.cot : (typeof m.overall === "number" ? m.overall : null));
  // best
  const candidates = [];
  if (typeof m.direct === "number") candidates.push(m.direct);
  if (typeof m.cot === "number") candidates.push(m.cot);
  if (typeof m.overall === "number") candidates.push(m.overall);
  return candidates.length ? Math.max(...candidates) : null;
};

let sortKey = "best";
let sortDir = "desc";

const lbBody = document.getElementById("lbBody");
const lbMetric = document.getElementById("lbMetric");
const lbSource = document.getElementById("lbSource");
const lbSearch = document.getElementById("lbSearch");
const lbTable = document.getElementById("lbTable");

function filterLeaderboard() {
  const metric = lbMetric ? lbMetric.value : "best";
  const source = lbSource ? lbSource.value : "all";
  const q = lbSearch ? lbSearch.value.trim().toLowerCase() : "";

  return leaderboard
    .filter((m) => {
      if (source === "open" && m.source.toLowerCase() !== "open") return false;
      if (source === "closed" && m.source.toLowerCase() !== "closed") return false;
      if (q && !m.model.toLowerCase().includes(q)) return false;
      return true;
    })
    .map((m) => ({ ...m, _best: getVal(m, metric) }));
}

function sortRows(rows) {
  const key = sortKey;
  const dir = sortDir === "asc" ? 1 : -1;

  const getSort = (m) => {
    if (key === "model") return m.model.toLowerCase();
    if (key === "source") return m.source.toLowerCase();
    if (key === "direct") return (typeof m.direct === "number" ? m.direct : -Infinity);
    if (key === "cot") return (typeof m.cot === "number" ? m.cot : -Infinity);
    if (key === "best") return (typeof getVal(m, "best") === "number" ? getVal(m, "best") : -Infinity);
    return (typeof m._best === "number" ? m._best : -Infinity);
  };

  return rows.slice().sort((a, b) => {
    const A = getSort(a);
    const B = getSort(b);
    if (typeof A === "string" && typeof B === "string") return A.localeCompare(B) * dir;
    return (A - B) * dir;
  });
}

function renderLeaderboard() {
  if (!lbBody) return;
  const metric = lbMetric ? lbMetric.value : "best";

  let rows = filterLeaderboard();
  rows = sortRows(rows);

  // ranking based on current metric sorting (best by default)
  lbBody.innerHTML = rows.map((m, idx) => {
    const best = getVal(m, "best");
    const d = (typeof m.direct === "number" ? m.direct : null);
    const c = (typeof m.cot === "number" ? m.cot : null);

    const badge = m.source.toLowerCase() === "closed"
      ? `<span class="badge-closed">● Closed</span>`
      : `<span class="badge-open">● Open</span>`;

    const bestCls = metric === "best" ? "best" : "num";
    return `
      <tr>
        <td class="num"><b>${idx + 1}</b></td>
        <td><b>${m.model}</b></td>
        <td>${badge}</td>
        <td class="${d === null ? "num miss" : "num"}">${fmt(d)}</td>
        <td class="${c === null ? "num miss" : "num"}">${fmt(c)}</td>
        <td class="${best === null ? "num miss" : bestCls}"><b>${fmt(best)}</b></td>
      </tr>
    `;
  }).join("");
}

function bindSort() {
  if (!lbTable) return;
  const ths = lbTable.querySelectorAll("thead th[data-sort]");
  ths.forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.getAttribute("data-sort");
      if (!key) return;
      if (sortKey === key) sortDir = (sortDir === "asc" ? "desc" : "asc");
      else { sortKey = key; sortDir = "desc"; }
      renderLeaderboard();
    });
  });
}

if (lbMetric) lbMetric.addEventListener("change", () => { sortKey = lbMetric.value; renderLeaderboard(); });
if (lbSource) lbSource.addEventListener("change", renderLeaderboard);
if (lbSearch) lbSearch.addEventListener("input", renderLeaderboard);

bindSort();
renderLeaderboard();
