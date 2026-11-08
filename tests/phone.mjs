// The family phone harness, rewritten for this piece (the 11/6 harness checks, same devices):
// iPhone 13 (390x664), iPhone SE (320x568), iPhone 13 on its side (844x390) with iOS safe-area
// insets, plus 375x812 and 1440x900. For each: 0 console errors, no horizontal scroll, Write on
// the first screen, "How it works" opens from a tap 11 px above its text and closes from a tap
// 15 px off the x's centre, every control reachable by Tab with a visible focus ring. Then
// reduced motion: no stroke animation, the finished circle and the scrubber at the last pass.
//
//   PUPPETEER=/path/to/node_modules/puppeteer-core node tests/phone.mjs http://127.0.0.1:8814/index.html [shots-dir]
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const puppeteer = require(process.env.PUPPETEER || "puppeteer-core");
const URL_ = process.argv[2], SHOTS = process.argv[3];
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const devices = [
  { name: "iphone13", w: 390, h: 664, dpr: 3, touch: true, insets: { top: 47, bottom: 34, left: 0, right: 0 } },
  { name: "iphoneSE", w: 320, h: 568, dpr: 2, touch: true, insets: { top: 20, bottom: 0, left: 0, right: 0 } },
  { name: "landscape", w: 844, h: 390, dpr: 3, touch: true, insets: { top: 0, bottom: 21, left: 47, right: 47 } },
  { name: "m375", w: 375, h: 812, dpr: 2, touch: true, insets: { top: 0, bottom: 0, left: 0, right: 0 } },
  { name: "d1440", w: 1440, h: 900, dpr: 1, touch: false, insets: { top: 0, bottom: 0, left: 0, right: 0 } },
];
const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new", args: ["--use-angle=metal", "--enable-gpu"] });
let fails = 0;
const check = (dev, name, ok, extra = "") => { if (!ok) fails++; console.log(`${ok ? "ok  " : "FAIL"} ${dev.padEnd(9)} ${name}${extra ? "  " + extra : ""}`); };

async function open(dev, reduced = false) {
  const page = await browser.newPage();
  const errors = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("requestfailed", (r) => { if (!/fonts\.(googleapis|gstatic)/.test(r.url())) errors.push("request failed " + r.url()); });
  await page.setViewport({ width: dev.w, height: dev.h, deviceScaleFactor: dev.dpr, isMobile: dev.touch, hasTouch: dev.touch });
  const cdp = await page.createCDPSession();
  try { await cdp.send("Emulation.setSafeAreaInsetsOverride", { insets: dev.insets }); } catch { /* older Chrome */ }
  if (reduced) await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
  await page.goto(URL_, { waitUntil: "load" });
  await page.waitForFunction(() => document.getElementById("write").getAttribute("aria-busy") === "false" && document.getElementById("stepout").textContent.startsWith(document.getElementById("step").max), { timeout: 120000 });
  return { page, errors };
}
const tapAt = async (page, dev, x, y) => dev.touch ? page.touchscreen.tap(x, y) : page.mouse.click(x, y);

for (const dev of devices) {
  const { page, errors } = await open(dev);
  const m = await page.evaluate(() => {
    const r = (id) => { const b = document.getElementById(id).getBoundingClientRect(); return { x: b.x, y: b.y, w: b.width, h: b.height, bottom: b.bottom, right: b.right }; };
    return { sw: document.documentElement.scrollWidth, iw: innerWidth, ih: innerHeight, write: r("write"), made: r("made") };
  });
  check(dev.name, "no horizontal scroll", m.sw <= m.iw, `${m.sw} <= ${m.iw}`);
  check(dev.name, "Write on the first screen", m.write.y >= 0 && m.write.bottom <= m.ih && m.write.right <= m.iw);
  await new Promise((r) => setTimeout(r, 400));  // let button colour transitions finish
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${dev.name}.png` });
  // How it works: tap 11 px above the text, then 15 px off the x's centre
  await tapAt(page, dev, m.made.x + m.made.w / 2, m.made.y - 11);
  await new Promise((r) => setTimeout(r, 450));
  const opened = await page.evaluate(() => document.getElementById("panel").classList.contains("on"));
  check(dev.name, "How it works opens from a tap 11 px above", opened);
  if (SHOTS && opened) await page.screenshot({ path: `${SHOTS}/${dev.name}-how.png` });
  const c = await page.evaluate(() => { const b = document.getElementById("close").getBoundingClientRect(); return { x: b.x + b.width / 2, y: b.y + b.height / 2 }; });
  await tapAt(page, dev, c.x - 15, c.y);
  await new Promise((r) => setTimeout(r, 450));
  check(dev.name, "panel closes from a tap 15 px off the x", await page.evaluate(() => !document.getElementById("panel").classList.contains("on")));
  // Compare runs to the end
  await page.click("#compare");
  await page.waitForFunction(() => document.getElementById("compare").getAttribute("aria-pressed") === "false", { timeout: 60000 });
  const twinDone = await page.evaluate(() => document.getElementById("twintxt").textContent.length > 100);
  check(dev.name, "Compare writes the twin's passage", twinDone);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${dev.name}-compare.png` });
  // keyboard
  if (!dev.touch) {
    const want = ["made", "step", "write", "compare", "remask", "k"];
    const seen = new Set(), noRing = [];
    await page.evaluate(() => document.activeElement && document.activeElement.blur());
    for (let i = 0; i < 12; i++) {
      await page.keyboard.press("Tab");
      const f = await page.evaluate(() => { const a = document.activeElement; const cs = getComputedStyle(a); return { id: a.id, ring: cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) > 0 }; });
      if (f.id) { seen.add(f.id); if (!f.ring) noRing.push(f.id); }
    }
    const missing = want.filter((id) => !seen.has(id));
    check(dev.name, "every control reachable by Tab", !missing.length, missing.join(","));
    check(dev.name, "focus ring visible", !noRing.length, noRing.join(","));
  }
  check(dev.name, "0 console errors", errors.length === 0, errors.slice(0, 3).join(" | "));
  await page.close();
}

// reduced motion: the finished circle, no stroke animation
{
  const dev = devices[4];
  const { page, errors } = await open(dev, true);
  const r = await page.evaluate(async () => {
    let frames = 0; const t0 = performance.now();
    await new Promise((res) => { const f = () => { frames++; if (performance.now() - t0 < 1000) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
    const a = document.getElementById("ring").toDataURL();
    await new Promise((res) => setTimeout(res, 500));
    return { same: a === document.getElementById("ring").toDataURL(), step: document.getElementById("step").value, max: document.getElementById("step").max };
  });
  check("reduced", "finished circle, scrubber at the last pass", r.step === r.max, `${r.step}/${r.max}`);
  check("reduced", "ring is still (no animation)", r.same);
  check("reduced", "0 console errors", errors.length === 0, errors.join(" | "));
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/reduced.png` });
  await page.close();
}
await browser.close();
console.log(fails ? `VERDICT: FAIL (${fails})` : "VERDICT: PASS");
process.exit(fails ? 1 : 0);
