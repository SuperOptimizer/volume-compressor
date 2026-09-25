// Headless check of dist/volcomp_viewer.html from file:// (or an http URL):
// opens the example volume + prediction via the URL hash, waits for chunks to
// stream and decode, screenshots the page, and prints in-browser checksums of
// given chunks (compare with ref_checksums.py). Needs playwright + chromium:
//   npm i playwright && npx playwright install chromium
//   NODE_PATH=<dir>/node_modules node web/test/browser_check.cjs [page-url] [out-dir]
const path = require("path");
const { chromium } = require("playwright");

const B = "https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHerc0343P";
const VOL = B + "/volumes/20250521134555-8.640um-1.2m-116keV-masked.zarr";
const PRED = B + "/representations/predictions/surfaces/20250521134555-surface-20260925170000-surface-m7-L0-prob.zarr";
const page0 = process.argv[2] || "file://" + path.resolve(__dirname, "../dist/volcomp_viewer.html");
const outDir = process.argv[3] || ".";

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const logs = [];
  page.on("console", (m) => logs.push(m.type() + ": " + m.text()));
  page.on("pageerror", (e) => logs.push("pageerror: " + e.message));
  const url = page0 + "#vol=" + encodeURIComponent(VOL) + "&pred=" + encodeURIComponent(PRED) + "&c=2700,2528,2528";
  const t0 = Date.now();
  await page.goto(url);
  // wait until the queue drains (or 90 s)
  let st;
  for (let i = 0; i < 180; i++) {
    await page.waitForTimeout(500);
    st = await page.evaluate(() => ({
      decoded: vcViewer.stats.decoded, bytes: vcViewer.stats.bytes, req: vcViewer.stats.requests,
      status: document.getElementById("status-main").textContent, warn: document.getElementById("status-warn").textContent,
      busy: document.getElementById("status-main").textContent.match(/queue (\d+) · fetching (\d+) · decoding (\d+)/),
    }));
    if (st.decoded > 0 && st.busy && st.busy.slice(1).every((x) => x === "0")) break;
  }
  console.log("settled after", ((Date.now() - t0) / 1000).toFixed(1), "s:", st.status, "|", st.warn);
  await page.screenshot({ path: path.join(outDir, "viewer_overview.png") });
  // zoom the z view to 1:1 around the cursor at level 0 and let it load
  await page.evaluate(() => { const V = vcViewer.S.views[0]; V.zoom = 1; V.center = [2528, 2528]; });
  await page.mouse.move(400, 300);
  await page.keyboard.press("w");
  for (let i = 0; i < 120; i++) {
    await page.waitForTimeout(500);
    const b = await page.evaluate(() => document.getElementById("status-main").textContent.match(/queue (\d+) · fetching (\d+) · decoding (\d+)/));
    if (b && b.slice(1).every((x) => x === "0")) break;
  }
  await page.screenshot({ path: path.join(outDir, "viewer_zoom.png") });
  // deblock on, threshold mask on
  await page.keyboard.press("d");
  await page.keyboard.press("t");
  await page.waitForTimeout(6000);
  await page.screenshot({ path: path.join(outDir, "viewer_smooth_mask.png") });
  const checks = [
    [VOL + "/0", 22, 16, 16, 0, 0], [VOL + "/0", 22, 16, 16, 2.0, 3],
    [PRED + "/8.64", 20, 16, 20, 0, 0], [PRED + "/8.64", 20, 16, 20, 0.6, 1],
  ];
  for (const c of checks) {
    const r = await page.evaluate((c) => vcViewer.chunkChecksum(...c), c);
    console.log("chunk", c.slice(1).join(","), c[0].split("/").slice(-2).join("/"), JSON.stringify(r));
  }
  const final = await page.evaluate(() => document.getElementById("status-main").textContent);
  console.log("final:", final);
  const shown = logs.filter((l) => !/status of 404/.test(l));
  console.log(`console: ${logs.length - shown.length} 404 lines (missing shards = air)`);
  if (shown.length) console.log(shown.join("\n"));
  // phase 2: the pickers, from a page without a hash (mirror index -> sample -> volume -> overlay),
  // on the 2.215 um volume with the upstream 8.86 um threshold mask (overlay level 0 = CT level 2)
  const p2 = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  p2.on("pageerror", (e) => console.log("pageerror: " + e.message));
  await p2.goto(page0);
  await p2.waitForFunction(() => document.querySelectorAll("#sample option").length > 30, null, { timeout: 30000 });
  const nSamples = await p2.evaluate(() => document.querySelectorAll("#sample option").length - 1);
  await p2.selectOption("#sample", "PHerc0343P");
  await p2.waitForFunction(() => document.querySelectorAll("#volume option").length > 1 && window.vcViewer.S.ct, null, { timeout: 30000 });
  const vols = await p2.evaluate(() => [...document.querySelectorAll("#volume option")].map((o) => o.textContent));
  const preds = await p2.evaluate(() => [...document.querySelectorAll("#pred option")].map((o) => o.textContent));
  console.log("samples:", nSamples, "volumes:", vols.join(" | "), "preds:", preds.join(" | "));
  const v2 = await p2.evaluate(() => [...document.querySelectorAll("#volume option")].find((o) => /2\.215um/.test(o.value)).value);
  await p2.selectOption("#volume", v2);
  await p2.waitForFunction((v) => window.vcViewer.S.ct && window.vcViewer.S.ct.url === v, v2, { timeout: 30000 });
  const pr = await p2.evaluate(() => [...document.querySelectorAll("#pred option")].find((o) => /th0\.2/.test(o.value)).value);
  await p2.selectOption("#pred", pr);
  await p2.waitForFunction(() => window.vcViewer.S.pred, null, { timeout: 30000 });
  for (let i = 0; i < 120; i++) {
    await p2.waitForTimeout(500);
    const b = await p2.evaluate(() => document.getElementById("status-main").textContent.match(/queue (\d+) · fetching (\d+) · decoding (\d+)/));
    if (b && b.slice(1).every((x) => x === "0")) break;
  }
  const lab = await p2.evaluate(() => [...document.querySelectorAll(".vlabel")].map((e) => e.textContent));
  console.log("2.2um views:", lab.join(" || "));
  console.log("2.2um status:", await p2.evaluate(() => document.getElementById("status-main").textContent));
  await p2.screenshot({ path: path.join(outDir, "viewer_2um_th02.png") });
  // phase 3: mouse and keyboard on the z view (top-left quadrant)
  const box = await p2.evaluate(() => { const r = document.getElementById("view0").getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; });
  const cx = box[0] + box[2] / 2, cy = box[1] + box[3] / 2;
  const snap = () => p2.evaluate(() => { const S = window.vcViewer.S, V = S.views[0];
    return { z: S.cursor[0], y: S.cursor[1], zoom: V.zoom, c: V.center.slice(), level: S.level, win: S.win, max: S.maximized, smooth: S.smooth.on }; });
  const a0 = await snap();
  await p2.mouse.move(cx, cy);
  await p2.mouse.wheel(0, 100); await p2.waitForTimeout(200);
  const a1 = await snap();
  await p2.keyboard.down("Control"); await p2.mouse.wheel(0, -300); await p2.keyboard.up("Control"); await p2.waitForTimeout(200);
  const a2 = await snap();
  await p2.mouse.down(); await p2.mouse.move(cx + 80, cy + 40, { steps: 5 }); await p2.mouse.up(); await p2.waitForTimeout(200);
  const a3 = await snap();
  await p2.mouse.click(cx - 50, cy - 50); await p2.waitForTimeout(200);
  const a4 = await snap();
  await p2.mouse.move(cx, cy); await p2.mouse.down({ button: "right" }); await p2.mouse.move(cx - 60, cy, { steps: 4 }); await p2.mouse.up({ button: "right" });
  const a5 = await snap();
  await p2.keyboard.press(","); await p2.keyboard.press("f"); await p2.keyboard.press("d"); await p2.waitForTimeout(300);
  const a6 = await snap();
  await p2.keyboard.press("Escape"); await p2.keyboard.press("a");
  const ok = (c, m) => console.log((c ? "PASS " : "FAIL ") + m);
  ok(a1.z !== a0.z, `wheel steps the z slice (${a0.z.toFixed(1)} -> ${a1.z.toFixed(1)})`);
  ok(a2.zoom > a0.zoom, `ctrl+wheel zooms (${a0.zoom.toFixed(3)} -> ${a2.zoom.toFixed(3)})`);
  ok(a3.c[0] < a2.c[0] && a3.c[1] < a2.c[1], "drag pans");
  ok(a4.y !== a3.y && a4.z === a3.z, "click moves the cursor in-plane");
  ok(a5.win < a4.win, `right-drag changes the window (${a4.win} -> ${a5.win})`);
  ok(a6.level !== "auto" && a6.max === 0 && a6.smooth, `keys: level ${a6.level}, maximised ${a6.max}, deblock ${a6.smooth}`);
  await browser.close();
})();
