"""Web search and fetch tools (uses Playwright + Bing)."""
from __future__ import annotations

from agi_kit.tools.base import tool


@tool(
    "web_search",
    "Search the web using Bing. Returns top N titles + urls + snippets.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
    tags=["network"],
)
def web_search(query: str, top_k: int = 5) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "error: playwright not installed. pip install playwright; playwright install chromium"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
            })
            page.goto("https://www.bing.com/search?q=" + query.replace(" ", "+"), timeout=20000)
            page.wait_for_timeout(2500)
            results = page.locator("li.b_algo").all()
            out = []
            for r in results[:top_k]:
                try:
                    t = r.locator("h2 a").first.text_content(timeout=500).strip()
                except Exception:
                    t = ""
                try:
                    u = r.locator("h2 a").first.get_attribute("href", timeout=500) or ""
                except Exception:
                    u = ""
                try:
                    s = r.locator(".b_caption p, p").first.text_content(timeout=500).strip()
                except Exception:
                    s = ""
                if t or u:
                    out.append(f"- {t}\n  {u}\n  {s[:200]}".strip())
            browser.close()
            return "\n\n".join(out) or "(no results)"
    except Exception as e:
        return f"error: {e}"[:300]


@tool(
    "web_fetch",
    "Fetch the plain text content of a URL",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "default": 3000},
        },
        "required": ["url"],
    },
    tags=["network"],
)
def web_fetch(url: str, max_chars: int = 3000) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "error: playwright not installed"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            page.wait_for_timeout(1000)
            txt = page.evaluate("() => document.body.innerText") or ""
            browser.close()
            return txt.strip()[:max_chars] or "(empty)"
    except Exception as e:
        return f"error: {e}"[:300]