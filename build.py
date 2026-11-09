"""Build index.html (the page) and weights.js (the weights) from src/index.src.html, src/model.js and the exported weights.

    python build.py

Inputs: weights/{diffusion,gpt}.{q8,f32}, weights/stats.json (train/export.py),
train/logs/{diffusion,gpt}.json (train/train.py), weights/wordrate.json (tests/wordrate_all.mjs).
"""
import base64, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
W = ROOT / "weights"

stats = json.loads((W / "stats.json").read_text())
for kind in ("diffusion", "gpt"):
    log = json.loads((ROOT / "train" / "logs" / f"{kind}.json").read_text())
    stats[kind]["seconds"] = log["seconds"]
    stats[kind]["steps"] = log["steps"]
import os
assert os.environ.get("DRAFT") or stats["diffusion"]["steps"] == stats["gpt"]["steps"]
stats["train"] = {"steps": stats["gpt"]["steps"], "device": "Apple M5 Pro (PyTorch, MPS)"}
stats["wordrate"] = json.loads((W / "wordrate.json").read_text())
text = (ROOT / "data" / "input.txt").read_text()
stats["chars"] = sorted(set(text))

page = (ROOT / "src" / "index.src.html").read_text()
model_js = (ROOT / "src" / "model.js").read_text()
assert "</script" not in model_js
weights = {
    "w-dq": base64.b64encode((W / "diffusion.q8").read_bytes()).decode(),
    "w-df": base64.b64encode((W / "diffusion.f32").read_bytes()).decode(),
    "w-gq": base64.b64encode((W / "gpt.q8").read_bytes()).decode(),
    "w-gf": base64.b64encode((W / "gpt.f32").read_bytes()).decode(),
}
(ROOT / "weights.js").write_text("window.HEPTAPOD_WEIGHTS = " + json.dumps(weights) + ";\n")
subs = {
    "{{MODEL_JS}}": model_js,
    "{{STATS_JSON}}": json.dumps(stats, separators=(",", ":")).replace("</", "<\\/"),
}
for k, v in subs.items():
    assert page.count(k) == 1, k
    page = page.replace(k, v)
(ROOT / "index.html").write_text(page)
print("index.html", len(page.encode()), "bytes; weights.js", (ROOT / "weights.js").stat().st_size, "bytes")
