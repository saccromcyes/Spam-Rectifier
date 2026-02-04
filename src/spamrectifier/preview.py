"""Preview UI served from pure Python."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler


PREVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SpamRectifier Preview</title>
    <style>
      :root {
        color-scheme: light dark;
        --bg: #0b1020;
        --card: #141a2e;
        --text: #eef1ff;
        --muted: #a9b2d6;
        --accent: #6aa6ff;
        --accent-2: #59e1c9;
        --warning: #ffb86b;
        --glow: rgba(106, 166, 255, 0.35);
        font-family: "Inter", system-ui, -apple-system, sans-serif;
      }
      body {
        margin: 0;
        padding: 32px;
        background: radial-gradient(circle at top, #19203a, var(--bg));
        color: var(--text);
      }
      .container {
        max-width: 1040px;
        margin: 0 auto;
        display: grid;
        gap: 24px;
      }
      header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
      }
      .pill {
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(106, 166, 255, 0.2);
        color: var(--accent);
        font-size: 0.85rem;
      }
      .hero {
        background: linear-gradient(135deg, #1a2450, #16204a);
        border-radius: 18px;
        padding: 28px;
        display: grid;
        gap: 16px;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
      }
      .card {
        background: var(--card);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 0 24px rgba(0, 0, 0, 0.2);
      }
      textarea {
        width: 100%;
        min-height: 120px;
        background: rgba(7, 10, 22, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: var(--text);
        border-radius: 12px;
        padding: 12px;
        font-size: 0.95rem;
      }
      button {
        background: linear-gradient(135deg, var(--accent), #4a6bff);
        color: white;
        border: none;
        padding: 12px 18px;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 0 18px var(--glow);
      }
      button.secondary {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.24);
        color: var(--text);
        box-shadow: none;
      }
      pre {
        background: rgba(7, 10, 22, 0.7);
        padding: 14px;
        border-radius: 12px;
        overflow: auto;
        font-size: 0.85rem;
        color: var(--muted);
      }
      .metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
      }
      .metric {
        background: rgba(89, 225, 201, 0.12);
        padding: 12px;
        border-radius: 12px;
        color: var(--accent-2);
        font-weight: 600;
      }
      .banner {
        background: rgba(255, 184, 107, 0.15);
        color: var(--warning);
        padding: 10px 14px;
        border-radius: 12px;
        font-size: 0.85rem;
      }
      .signal {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .signal span {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 12px var(--glow);
        animation: pulse 2s infinite;
      }
      @keyframes pulse {
        0% { transform: scale(0.9); opacity: 0.7; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.7; }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <header>
        <div>
          <div class="pill">Preview Mode</div>
          <h1>SpamRectifier Control Center</h1>
          <p>Polished demo UI that mirrors the live API experience.</p>
        </div>
        <div class="card">
          <div class="signal"><span></span><strong>Live preview available</strong></div>
          <p class="muted">Connect to the API for real predictions.</p>
        </div>
      </header>
      <section class="hero">
        <h2>Try the classifier</h2>
        <textarea id="message" placeholder="Paste an email or SMS message...">Claim your exclusive reward now</textarea>
        <div class="grid">
          <button id="predict-btn">Predict</button>
          <button id="explain-btn" class="secondary">Explain</button>
        </div>
        <pre id="output">Waiting for input...</pre>
        <div class="banner">Tip: Run the FastAPI service to connect live predictions.</div>
      </section>
      <section class="grid">
        <div class="card">
          <h3>Model card snapshot</h3>
          <pre id="model-card">Model card unavailable (preview mode).</pre>
        </div>
        <div class="card">
          <h3>Key metrics</h3>
          <div class="metrics">
            <div class="metric">Precision: 0.92</div>
            <div class="metric">Recall: 0.89</div>
            <div class="metric">F1: 0.90</div>
            <div class="metric">Accuracy: 0.94</div>
          </div>
          <p class="muted">Replace with live metrics once connected to the API.</p>
        </div>
      </section>
    </div>
    <script>
      const output = document.getElementById("output");

      function mockResponse(type) {
        if (type === "predict") {
          return {
            prediction: "spam",
            probabilities: {spam: 0.93, ham: 0.07},
          };
        }
        return {
          prediction: "spam",
          probabilities: {spam: 0.93, ham: 0.07},
          top_tokens: [
            {token: "reward", contribution: 0.73},
            {token: "exclusive", contribution: 0.52},
            {token: "claim", contribution: 0.41},
          ],
        };
      }

      document.getElementById("predict-btn").addEventListener("click", () => {
        output.textContent = JSON.stringify(mockResponse("predict"), null, 2);
      });

      document.getElementById("explain-btn").addEventListener("click", () => {
        output.textContent = JSON.stringify(mockResponse("explain"), null, 2);
      });
    </script>
  </body>
</html>
"""


def load_preview_html() -> str:
    return PREVIEW_HTML


class PreviewRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path not in ("/", "/index.html"):
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        body = load_preview_html().encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return
