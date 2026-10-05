(function () {
  "use strict";

  function addCopyButtons() {
    var blocks = document.querySelectorAll(".post-content div.highlighter-rouge, .post-content figure.highlight");
    blocks.forEach(function (block) {
      if (block.classList.contains("language-mermaid")) return;
      var code = block.querySelector("pre code") || block.querySelector("pre");
      if (!code) return;
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy-btn";
      btn.textContent = "Copy";
      btn.addEventListener("click", function () {
        navigator.clipboard.writeText(code.innerText.replace(/\n$/, "")).then(function () {
          btn.textContent = "Copied";
          setTimeout(function () { btn.textContent = "Copy"; }, 1500);
        });
      });
      block.appendChild(btn);
    });
  }

  // ```mermaid fenced blocks -> rendered diagrams (loaded only when a page uses them)
  function renderMermaid() {
    var blocks = document.querySelectorAll(".post-content .language-mermaid");
    if (!blocks.length) return;
    blocks.forEach(function (block) {
      var src = (block.querySelector("code") || block).textContent;
      var div = document.createElement("div");
      div.className = "mermaid-diagram";
      var pre = document.createElement("pre");
      pre.className = "mermaid";
      pre.textContent = src;
      div.appendChild(pre);
      block.replaceWith(div);
    });
    import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs").then(function (m) {
      var mermaid = m.default;
      mermaid.initialize({
        startOnLoad: false,
        theme: "base",
        fontFamily: '"DM Sans", Arial, sans-serif',
        themeVariables: {
          background: "#f0eee6",
          primaryColor: "#faf9f5",
          primaryBorderColor: "#141413",
          primaryTextColor: "#141413",
          lineColor: "#3d3d3a",
          secondaryColor: "#e3dacc",
          tertiaryColor: "#e8e6dc"
        }
      });
      mermaid.run({ querySelector: ".mermaid" });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderMermaid();
    addCopyButtons();
  });
})();
