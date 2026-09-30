"use strict";

const TASKS = {
  qa: { endpoint: "/qa", field: "question", emptyMsg: "Please type a question first." },
  explain: { endpoint: "/explain", field: "topic", emptyMsg: "Please type a concept to explain." },
  summarize: { endpoint: "/summarize", field: "text", emptyMsg: "Please paste some text to summarize." },
  quiz: { endpoint: "/quiz", field: "text", emptyMsg: "Please type a topic or paste a passage." },
  recommend: { endpoint: "/learn/recommendations", field: "topic", emptyMsg: "Please type a subject you want to learn." },
};
const REQUEST_TIMEOUT_MS = 180000;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showMessage(box, text, className) {
  box.replaceChildren(el("p", className, text));
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inlineMd(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
}

function renderMarkdown(text) {
  const lines = escapeHtml(text).split(/\r?\n/);
  let html = "";
  let list = null;
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };

  for (const raw of lines) {
    const line = raw.trim();
    let m;
    if (!line) { closeList(); continue; }
    if ((m = line.match(/^#{1,6}\s+(.*)$/))) {
      closeList();
      html += `<h3>${inlineMd(m[1])}</h3>`;
    } else if ((m = line.match(/^[-*•]\s+(.*)$/))) {
      if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; }
      html += `<li>${inlineMd(m[1])}</li>`;
    } else if ((m = line.match(/^\d+[.)]\s+(.*)$/))) {
      if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; }
      html += `<li>${inlineMd(m[1])}</li>`;
    } else {
      closeList();
      html += `<p>${inlineMd(line)}</p>`;
    }
  }
  closeList();
  return html;
}

function renderText(box, text, markdown) {
  const out = el("div", "text-result");
  if (markdown) out.innerHTML = renderMarkdown(text);
  else out.textContent = text;
  box.replaceChildren(out);
}

function renderQuiz(box, quiz) {
  const wrap = el("div", "quiz");
  const score = el("p", "quiz-score muted", `Answered 0 of ${quiz.length}`);
  let answered = 0;
  let correct = 0;

  quiz.forEach((q, qi) => {
    const block = el("div", "quiz-question");
    block.appendChild(el("h3", null, `Question ${qi + 1}`));
    block.appendChild(el("p", "question-text", q.question));
    const options = el("div", "options");
    const feedback = el("p", "feedback");

    q.options.forEach((opt, oi) => {
      const btn = el("button", "option", `${String.fromCharCode(65 + oi)}. ${opt}`);
      btn.type = "button";
      btn.addEventListener("click", () => {
        options.querySelectorAll("button").forEach((b, bi) => {
          b.disabled = true;
          if (q.options[bi] === q.answer) b.classList.add("correct");
        });
        answered += 1;
        if (opt === q.answer) {
          correct += 1;
          feedback.textContent = "✓ Correct!";
          feedback.className = "feedback ok";
        } else {
          btn.classList.add("wrong");
          feedback.textContent = `✗ Not quite. The correct answer is: ${q.answer}`;
          feedback.className = "feedback bad";
        }
        score.textContent = answered === quiz.length
          ? `Score: ${correct} / ${quiz.length}`
          : `Answered ${answered} of ${quiz.length}`;
      });
      options.appendChild(btn);
    });

    block.append(options, feedback);
    wrap.appendChild(block);
  });
  wrap.appendChild(score);
  box.replaceChildren(wrap);
}

function renderLearningPath(box, path) {
  const wrap = el("div", "path");
  wrap.appendChild(el("h3", "path-title", `Learning path: ${path.topic}`));
  if (path.overview) wrap.appendChild(el("p", null, path.overview));

  path.stages.forEach((stage, i) => {
    const card = el("div", `stage stage-${stage.level.toLowerCase()}`);
    const head = el("div", "stage-head");
    head.append(el("span", "badge", `${i + 1}. ${stage.level}`), el("span", "duration", `⏱ ${stage.duration}`));
    card.appendChild(head);
    if (stage.goal) card.appendChild(el("p", "goal", stage.goal));

    const addList = (title, items, ordered) => {
      if (!items.length) return;
      card.appendChild(el("h4", null, title));
      const list = el(ordered ? "ol" : "ul");
      items.forEach((item) => list.appendChild(el("li", null, item)));
      card.appendChild(list);
    };
    addList("Topics (in order)", stage.topics, true);
    addList("Steps", stage.steps, false);
    addList("Resources", stage.resources.map((r) => `[${r.type}] ${r.title}`), false);
    wrap.appendChild(card);
  });

  if (path.tips && path.tips.length) {
    wrap.appendChild(el("h4", null, "Study tips"));
    const tips = el("ul");
    path.tips.forEach((t) => tips.appendChild(el("li", null, t)));
    wrap.appendChild(tips);
  }
  wrap.appendChild(el("p", "muted small", "Resources are AI-suggested; search for them by name."));
  box.replaceChildren(wrap);
}

function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight + 2}px`;
  textarea.style.overflowY = textarea.scrollHeight > 320 ? "auto" : "hidden";
}

function setupSection(section) {
  const taskKey = section.dataset.task;
  const task = TASKS[taskKey];
  const limit = Number(section.dataset.limit);
  const form = section.querySelector("form");
  const input = form.querySelector("input, textarea");
  const button = form.querySelector("button");
  const box = section.querySelector(".result-box");
  const label = button.textContent;
  let busy = false;

  function setBusy(state) {
    busy = state;
    button.disabled = state;
    button.textContent = state ? "Working…" : label;
    section.setAttribute("aria-busy", String(state));
  }

  async function submit() {
    if (busy) return;
    const value = input.value.trim();
    if (!value) {
      showMessage(box, task.emptyMsg, "error");
      input.focus();
      return;
    }
    if (value.length > limit) {
      showMessage(box, `Your input is too long (${value.length.toLocaleString()} characters). The maximum is ${limit.toLocaleString()}.`, "error");
      return;
    }

    setBusy(true);
    box.replaceChildren(el("div", "loading", "Thinking… please wait."));
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch(task.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [task.field]: value }),
        signal: controller.signal,
      });
      let data = null;
      try { data = await response.json(); } catch (_) {}

      if (!response.ok || !data || !data.success) {
        showMessage(box, (data && data.error) || `Request failed (HTTP ${response.status}). Please try again.`, "error");
        return;
      }
      if (taskKey === "quiz") renderQuiz(box, data.quiz);
      else if (taskKey === "recommend") renderLearningPath(box, data.learning_path);
      else renderText(box, data.result, taskKey !== "explain");
    } catch (err) {
      showMessage(
        box,
        err.name === "AbortError"
          ? "The request timed out. Please try again."
          : "Network error: could not reach the EduGenie server. Is it running?",
        "error"
      );
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });

  if (input.tagName === "TEXTAREA") {
    input.addEventListener("input", () => autoGrow(input));
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        submit();
      }
    });
  }
}

document.querySelectorAll("section.tool").forEach(setupSection);
