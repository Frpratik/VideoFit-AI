"""Optional developer tool: drive the running app with a real browser and save the README screenshots.

Needs the server running (uvicorn app.api.main:app --port 8000), a valid GROQ_API_KEY, and
`pip install playwright` (it uses your installed Microsoft Edge or Chrome, no extra download).
It makes real model calls (about 20,000 tokens).
"""

import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(os.getenv("VF_OUT") or Path(__file__).resolve().parent.parent / "docs" / "screenshots")
BASE = os.getenv("VF_BASE", "http://localhost:8000")
COMMENTS = "\n".join([
    "Great explanation, the hill in the fog analogy finally made gradient descent click!",
    "Can you make a video on the math of backpropagation?",
    "The part about the learning rate was confusing for me",
    "Audio is a bit quiet in the second half",
    "Please add a coding example with PyTorch",
    "Best beginner lecture on neural networks I have seen",
])


def main(channel: str = "msedge") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel)
        page = browser.new_page(viewport={"width": 1360, "height": 860}, device_scale_factor=1)
        page.goto(BASE)

        page.click("#demo-btn")
        page.wait_for_function("document.getElementById('transcript').value.length > 100")
        page.locator("details:has(#comments)").evaluate("d => d.open = true")
        page.fill("#comments", COMMENTS)
        page.screenshot(path=str(OUT / "01-load.png"))

        page.click("#analyze-btn")
        page.wait_for_function("!document.getElementById('dash').hidden", timeout=300_000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "02-dashboard-top.png"))
        page.locator("#dash section:has(.reel)").scroll_into_view_if_needed()
        page.evaluate("window.scrollBy(0, -20)")
        page.screenshot(path=str(OUT / "03-dashboard-evidence.png"))
        page.locator("#dash").screenshot(path=str(OUT / "04-dashboard-full.png"))

        page.click('.tab[data-view="chat"]')
        for question in ("How does a single neuron compute its output?", "Now explain that in simple Hindi"):
            before = page.locator(".msg.bot").count()
            page.fill("#question", question)
            page.click("#send-btn")
            page.wait_for_function(
                "n => document.querySelectorAll('.msg.bot').length > n && !document.querySelector('.dots')",
                arg=before, timeout=120_000)
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "05-chat.png"))
        browser.close()
    print("saved to", OUT)


if __name__ == "__main__":
    main(*sys.argv[1:2])
