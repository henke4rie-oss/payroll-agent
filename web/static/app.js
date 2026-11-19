// Penny's chat page. Plain ES module, no build step.
// Safety rule: text from the model is only ever added with textContent / text nodes, never as HTML.

const SUGGESTIONS = [
  "Hi Penny",
  "Tax numbers for everyone",
  "What is Thandi Nkosi's net pay?",
  "Change Thandy's age to 66",
  "Change Thandi Nkosi's age to 66",
  "What would Sipho Khumalo's pay be at 66?",
  "What's the total payroll?",
];

const WELCOME = "Ask me about pay or tax, or pick a suggestion below.";

const messagesEl = document.getElementById("messages");
const chipsEl = document.getElementById("chips");
const form = document.getElementById("composer");
const input = document.getElementById("composer-input");
const sendBtn = document.getElementById("send");
const themeBtn = document.getElementById("theme-toggle");
const newChatBtn = document.getElementById("new-chat");

let busy = false;
let openCard = null; // the approval card that is still waiting for a click, if any

// ---------- small helpers ----------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function money(value) {
  return "R" + Number(value).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function newSessionId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function loadSessionId() {
  try {
    const saved = localStorage.getItem("penny-session");
    if (saved) return saved;
  } catch { /* storage can be blocked; a fresh id per page load still works */ }
  return saveSessionId(newSessionId());
}

function saveSessionId(id) {
  try { localStorage.setItem("penny-session", id); } catch { /* ignore */ }
  return id;
}

let sessionId = loadSessionId();

function scrollToEnd() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ---------- safe rendering of the model's light markdown (**bold**, lists, line breaks) ----------

function addInline(parent, text) {
  for (const part of text.split(/(\*\*[^*]+\*\*)/g)) {
    if (!part) continue;
    if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) {
      parent.append(el("strong", "", part.slice(2, -2)));
    } else {
      parent.append(document.createTextNode(part));
    }
  }
}

function renderMarkdown(text) {
  const frag = document.createDocumentFragment();
  let list = null;
  let listKind = null;
  let paragraph = null;
  let mdTable = null; // a markdown table the model wrote: shown as a real table, still text-only
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const ordered = line.match(/^\d+\.\s+(.*)$/);
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    if (line.startsWith("|") && line.endsWith("|")) {
      list = null;
      paragraph = null;
      if (/^\|[\s:|-]+\|$/.test(line)) continue; // the |---|---| separator row
      const cells = line.slice(1, -1).split("|").map((c) => c.trim());
      if (!mdTable) {
        mdTable = el("table");
        const wrap = el("div", "figures");
        wrap.append(mdTable);
        frag.append(wrap);
        const headRow = el("tr");
        for (const c of cells) {
          const th = el("th");
          addInline(th, c);
          headRow.append(th);
        }
        mdTable.append(el("thead"), el("tbody"));
        mdTable.tHead.append(headRow);
      } else {
        const tr = el("tr");
        for (const c of cells) {
          const td = el("td");
          addInline(td, c);
          tr.append(td);
        }
        mdTable.tBodies[0].append(tr);
      }
      continue;
    }
    mdTable = null;
    if (ordered || bullet) {
      const kind = ordered ? "ol" : "ul";
      if (!list || listKind !== kind) {
        list = el(kind);
        listKind = kind;
        frag.append(list);
      }
      paragraph = null;
      const item = el("li");
      addInline(item, (ordered || bullet)[1]);
      list.append(item);
    } else if (!line) {
      list = null;
      paragraph = null;
    } else {
      list = null;
      if (!paragraph) {
        paragraph = el("p");
        frag.append(paragraph);
      } else {
        paragraph.append(el("br"));
      }
      addInline(paragraph, line);
    }
  }
  return frag;
}

// ---------- figures tables: built from the tool results the server sends, not from the model's words ----------

function table(headers, rows) {
  const t = el("table");
  const head = el("thead");
  const headRow = el("tr");
  for (const h of headers) headRow.append(el("th", "", h));
  head.append(headRow);
  const body = el("tbody");
  for (const row of rows) {
    const tr = el("tr");
    for (const cell of row) tr.append(el("td", "", cell));
    body.append(tr);
  }
  t.append(head, body);
  return t;
}

function figuresBox(title, tableEl) {
  const box = el("div", "figures");
  box.append(el("p", "figures-title", title), tableEl);
  return box;
}

function payslipBox(p) {
  const rows = [
    ["Gross pay", money(p.gross_monthly)],
    ["PAYE (tax)", money(p.paye)],
    ["UIF (employee)", money(p.uif_employee)],
    ["Net pay", money(p.net_pay)],
  ];
  if (p.sdl) rows.push(["SDL (paid by the employer)", money(p.sdl)]);
  return figuresBox("Monthly figures, from the tax engine", table(["", "Amount"], rows));
}

function teamBox(rows) {
  const show = (r, key) => (r[key] === undefined ? "n/a" : money(r[key]));
  return figuresBox(
    "Everyone, monthly, straight from the payroll engine",
    table(
      ["Name", "Age", "Gross", "PAYE", "UIF", "SDL", "Net pay"],
      rows.map((r) => [r.name, String(r.age), show(r, "gross"), show(r, "paye"), show(r, "uif"), show(r, "sdl"), show(r, "net")]),
    ),
  );
}

// ---------- "How Penny got this" ----------

function traceBox(lines) {
  const details = el("details", "trace");
  details.append(el("summary", "", "How Penny got this"));
  const list = el("ul");
  for (const line of lines) list.append(el("li", "", line));
  details.append(list);
  return details;
}

// ---------- messages ----------

function addMessage(role, bodyChildren, extraClass = "") {
  const row = el("div", `msg msg-${role} ${extraClass}`.trim());
  if (role === "penny") {
    const avatar = el("span", "avatar", "P");
    avatar.setAttribute("aria-hidden", "true");
    row.append(avatar);
  }
  const body = el("div", "msg-body");
  body.append(...bodyChildren);
  row.append(body);
  messagesEl.append(row);
  scrollToEnd();
  return row;
}

function bubble(content) {
  const b = el("div", "bubble");
  if (typeof content === "string") b.append(renderMarkdown(content));
  else b.append(content);
  return b;
}

function addUser(text) {
  addMessage("user", [bubble(el("p", "", text))]);
}

function showTyping() {
  const dots = el("div", "bubble");
  const inner = el("span", "dots");
  inner.setAttribute("aria-hidden", "true");
  inner.append(el("span"), el("span"), el("span"));
  dots.append(inner, el("span", "sr-only", "Penny is thinking"));
  return addMessage("penny", [dots]);
}

function summaryText(summary) {
  const p = el("p", "approval-text");
  const match = summary.match(/(R?[\d,]+(?:\.\d+)?)\s*->\s*(R?[\d,]+(?:\.\d+)?)/);
  if (!match) {
    p.textContent = summary;
    return p;
  }
  p.append(document.createTextNode(summary.slice(0, match.index)));
  const change = el("span", "change");
  change.append(document.createTextNode(match[1] + " → "), el("span", "after", match[2]));
  p.append(change, document.createTextNode(summary.slice(match.index + match[0].length)));
  return p;
}

function settleCard(card, className, statusText) {
  card.classList.add(className);
  card.querySelector(".approval-actions")?.remove();
  card.append(el("p", "approval-status", statusText));
  if (openCard === card) openCard = null;
}

function approvalCard(approval) {
  const card = el("div", "approval");
  card.tabIndex = -1;
  card.setAttribute("role", "group");
  card.setAttribute("aria-label", "Approval needed");
  const approve = el("button", "btn btn-primary", "Approve");
  const reject = el("button", "btn", "Reject");
  approve.type = reject.type = "button";
  const actions = el("div", "approval-actions");
  actions.append(approve, reject);
  card.append(el("p", "approval-title", "Approval needed"), summaryText(approval.summary), actions);

  const decide = async (yes) => {
    if (busy) return;
    approve.disabled = reject.disabled = true;
    setBusy(true);
    const typing = showTyping();
    try {
      const result = await post("/api/approve", { session_id: sessionId, approval_id: approval.id, approve: yes });
      typing.remove();
      if (!yes) settleCard(card, "approval-cancelled", "Cancelled · nothing was changed");
      else if (result.status === "approved") settleCard(card, "approval-done", "Approved · saved to the audit log");
      else settleCard(card, "approval-cancelled", "Not applied · nothing was changed");
      renderReply(result);
    } catch (err) {
      typing.remove();
      settleCard(card, "approval-cancelled", err.status === 404 ? "This request has expired · ask me again" : "Could not reach Penny · nothing was changed");
    } finally {
      setBusy(false);
      input.focus();
    }
  };
  approve.addEventListener("click", () => decide(true));
  reject.addEventListener("click", () => decide(false));
  return card;
}

function renderReply(reply) {
  const parts = [];
  if (reply.status === "needs_approval") {
    parts.push(bubble("I need your OK before I change anything."));
    const card = approvalCard(reply.approval);
    openCard = card;
    parts.push(card);
  } else {
    parts.push(bubble(reply.message));
    if (reply.payslip) parts.push(payslipBox(reply.payslip));
    if (reply.table) parts.push(teamBox(reply.table));
  }
  if (reply.trace && reply.trace.length) parts.push(traceBox(reply.trace));
  const row = addMessage("penny", parts, reply.status === "error" ? "msg-error" : "");
  if (reply.status === "needs_approval") openCard.focus();
  return row;
}

// ---------- talking to the server ----------

async function post(path, body) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) {
    const err = new Error("request failed");
    err.status = res.status;
    err.detail = data && typeof data.detail === "string" ? data.detail : "";
    throw err;
  }
  return data;
}

function setBusy(value) {
  busy = value;
  sendBtn.disabled = value;
  input.disabled = value;
  chipsEl.querySelectorAll("button").forEach((b) => (b.disabled = value));
}

async function send(text) {
  text = text.trim();
  if (!text || busy) return;
  if (openCard) settleCard(openCard, "approval-cancelled", "Replaced by your new message · nothing was changed");
  addUser(text);
  input.value = "";
  setBusy(true);
  const typing = showTyping();
  try {
    renderReply(await post("/api/chat", { session_id: sessionId, message: text }));
  } catch (err) {
    const message = err.status === 503 && err.detail ? err.detail : "I can't reach the server right now. Nothing was changed.";
    addMessage("penny", [bubble(message)], "msg-error");
  } finally {
    typing.remove();
    setBusy(false);
    if (!openCard) input.focus(); // an open approval card keeps the focus, so it is announced and reachable
  }
}

// ---------- setup ----------

function showWelcome() {
  addMessage("penny", [bubble(WELCOME)]);
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const toLight = theme === "dark";
  themeBtn.textContent = toLight ? "Light" : "Dark";
  themeBtn.setAttribute("aria-label", toLight ? "Switch to light theme" : "Switch to dark theme");
}

for (const text of SUGGESTIONS) {
  const chip = el("button", "chip", text);
  chip.type = "button";
  chip.addEventListener("click", () => send(text));
  chipsEl.append(chip);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  send(input.value);
});

themeBtn.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("penny-theme", next); } catch { /* ignore */ }
});

newChatBtn.addEventListener("click", () => {
  if (busy) return;
  sessionId = saveSessionId(newSessionId()); // a new session also gets a fresh copy of the demo data
  openCard = null;
  messagesEl.replaceChildren();
  showWelcome();
  input.focus();
});

try {
  const savedTheme = localStorage.getItem("penny-theme");
  if (savedTheme === "light" || savedTheme === "dark") applyTheme(savedTheme);
} catch { /* ignore */ }

showWelcome();
