// Headless check of dist/volcomp_viewer.html from file:// (or an http URL). Needs
// playwright + chromium, python3 with python/volcomp_zarr built, and the zstd CLI:
//   (cd web/.tools && npm i playwright && npx playwright install chromium)
//   NODE_PATH=web/.tools/node_modules node web/test/browser_check.cjs [page-url] [out-dir]
// Phases (each prints PASS/FAIL lines; the exit code is the number of FAILs):
//   1 the old #vol=&pred=&c= hash (8.64 um CT + m7 L0), chunk checksums vs python
//   2 CT 2.215 um + m7 -L2 probabilities: geometry, and composited tiles vs ref_composite.py
//   3 the L2 prediction alone (no CT)
//   4 two overlays (probabilities, additive + the th0.2 mask, categorical) vs ref_composite.py
//   5 URL-hash round trip after changing settings in the layer panel
//   6 pickers: sample -> volume -> add-layer panel (representations/ listing) -> add
//   7 mouse and keys
//   8 a region set (volcomp+zstd teacher regions): chunk checksum vs python, rendering
//   9 stability: 10 s idle after settling = 0 requests, decodes and redraws (3 layers, default and
//     small caches), the cache-share warning, zarr v2 rejection, renderer memory at 3 layers
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const { chromium } = require("playwright");

const M = "https://dl.ash2txt.org/community-uploads/forrest/volcomp";
const B = M + "/PHerc0343P";
const CT8 = B + "/volumes/20250521134555-8.640um-1.2m-116keV-masked.zarr";
const PR8 = B + "/representations/predictions/surfaces/20250521134555-surface-20260925170000-surface-m7-L0-prob.zarr";
const CT2 = B + "/volumes/20260304131111-2.215um-0.4m-111keV-masked.zarr";
const PR2 = B + "/representations/predictions/surfaces/20260304131111-surface-20260925170000-surface-m7-L2-prob.zarr";
const TH2 = B + "/representations/predictions/surfaces/20260304131111-surface-20260413222639-surface-m7-L2-th0.2.zarr";
const P4 = M + "/PHercParis4";
const CTP4 = P4 + "/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr";
const REG = P4 + "/representations/predictions/teacher_regions/recto-2.4um";
const P4PR = P4 + "/representations/predictions/surfaces/20260411134726-surface-20260925170000-surface-m7-L2-prob.zarr";
const P4TH = P4 + "/representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr";
const S3TH = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343P/representations/predictions/surfaces/20260304131111-surface-20260413222639-surface-m7-L2-th0.2.zarr";
const page0 = process.argv[2] || "file://" + path.resolve(__dirname, "../dist/volcomp_viewer.html");
const outDir = process.argv[3] || ".";
const PY = path.join(__dirname, "ref_composite.py"), PYC = path.join(__dirname, "ref_checksums.py");

let fails = 0;
const ok = (c, m) => { console.log((c ? "PASS " : "FAIL ") + m); if (!c) fails++; };
const hashOf = (layers, extra = "") => "#" + layers.map((l) => "l=" + encodeURIComponent(l)).join("&") + extra;
const py = (args) => JSON.parse(execFileSync("python3", args, { encoding: "utf8", maxBuffer: 1 << 24 }).trim().split("\n").pop());
function refTile(spec, state) {
  const f = path.join(outDir, "tile_spec.json");
  fs.writeFileSync(f, JSON.stringify(Object.assign({}, spec, { layers: state.map((l, i) => ({ url: l.url, level: spec.levels[i], opts: l.opts })) })));
  return py([PY, f]);
}

async function newPage(browser, hash, dsf, vw, vh) {
  const page = await browser.newPage({ viewport: { width: vw || 1600, height: vh || 1000 }, deviceScaleFactor: dsf || 1 });
  page.on("pageerror", (e) => { console.log("pageerror: " + e.message); fails++; });
  page.on("console", (m) => { if (m.type() === "error" && !/status of 404/.test(m.text())) console.log("console error: " + m.text()); });
  await page.goto(page0 + (hash || ""));
  await page.waitForFunction(() => window.vcViewer && window.vcViewer.ready, null, { timeout: 60000 });
  return page;
}
// wait until something decoded and the loader is idle (or maxS seconds)
async function settle(page, maxS = 90) {
  const t0 = Date.now();
  let st = "";
  for (let i = 0; i < maxS * 2; i++) {
    await page.waitForTimeout(500);
    st = await page.evaluate(() => document.getElementById("status-main").textContent);
    const m = st.match(/queue (\d+) · fetching (\d+) · decoding (\d+)/);
    if (m && m.slice(1).every((x) => x === "0") && /decoded [1-9]/.test(st) && i > 2) break;
  }
  return { s: ((Date.now() - t0) / 1000).toFixed(1), st };
}
const state = (page) => page.evaluate(() => window.vcViewer.layerState());
const labels = (page) => page.evaluate(() => [...document.querySelectorAll(".vlabel")].map((e) => e.textContent.replace(/\n/g, " | ")));
async function zoomZ(page, zoom, cu, cv) { // the z view: zoom (px per µm) about (x, y) = (cu, cv) µm
  await page.evaluate(([z, u, v]) => { const V = window.vcViewer.S.views[0]; V.zoom = z; V.center = [u, v]; window.vcViewer.requestRender(); }, [zoom, cu, cv]);
}

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch();
  const t00 = Date.now();

  // ---- 1: the old hash form
  {
    const page = await newPage(browser, "#vol=" + encodeURIComponent(CT8) + "&pred=" + encodeURIComponent(PR8) + "&c=2700,2528,2528");
    const r = await settle(page);
    console.log("[1] settled after", r.s, "s:", r.st);
    const ls = await state(page);
    ok(ls.length === 2 && ls[0].url === CT8 && ls[1].url === PR8, "old #vol=&pred= hash opens the CT and the overlay as two layers");
    ok(ls[1].p0[0] === 8.64 && ls[0].p0[0] === 8.64, `8.64 um CT and m7 L0 on one grid (${ls[0].p0[0]} / ${ls[1].p0[0]} um)`);
    const cu = await page.evaluate(() => window.vcViewer.S.cursor);
    ok(Math.abs(cu[0] - 2700.5 * 8.64) < 1e-6, `old c= (CT voxels) becomes micrometres (${cu[0].toFixed(2)})`);
    await page.screenshot({ path: path.join(outDir, "viewer_legacy.png") });
    for (const [lv, c, sm, fl, key] of [[CT8 + "/0", [22, 16, 16], 0, 0, "plain"], [CT8 + "/0", [22, 16, 16], 2.0, 3, "ct_smooth"], [PR8 + "/8.64", [20, 16, 20], 0, 0, "plain"]]) {
      const b = await page.evaluate(([u, c, s, f]) => window.vcViewer.chunkChecksum(u, ...c, s, f), [lv, c, sm, fl]);
      const p = py([PYC, "url", lv, ...c.map(String)]);
      ok(b.sha256 === p[key], `chunk ${c} of ${lv.split("/").slice(-2).join("/")} ${key}: browser ${String(b.sha256).slice(0, 16)} python ${String(p[key]).slice(0, 16)}`);
    }
    await page.close();
  }

  // ---- 2: CT 2.215 um + m7 L2 probabilities
  const P = 2.215;
  const cur = [10400.5 * P, 5328.5 * P, 7168.5 * P];
  const extra = `&cu=${cur.join(",")}`;
  {
    const page = await newPage(browser, hashOf([CT2, PR2], extra));
    const ls = await state(page);
    ok(ls.length === 2, "two layers from #l=&l=");
    ok(ls[1].rule.startsWith("source CT 2.215"), "m7 -L2 pitch rule: " + ls[1].rule);
    const same = ls[1].levels[0].pitch.every((p, a) => p === ls[0].levels[2].pitch[a]) &&
      ls[1].levels[0].shape.every((n, a) => n === ls[0].levels[2].shape[a]) && ls[1].levels[0].origin.every((o) => o === 0);
    ok(same, `m7 L2 level 0 = CT level 2 grid (pitch ${ls[1].levels[0].pitch[0]} vs ${ls[0].levels[2].pitch[0]}, shape ${ls[1].levels[0].shape} vs ${ls[0].levels[2].shape})`);
    await zoomZ(page, 1 / P, cur[2], cur[1]); // 1 screen px per CT voxel
    const r = await settle(page);
    console.log("[2] settled after", r.s, "s:", r.st);
    const lab = await labels(page);
    console.log("[2] views:", lab.join(" || "));
    ok(/2\.215um\S* L0 /.test(lab[0]) && /m7-L2-prob L0 8\.86/.test(lab[0]), "1 px per CT voxel: CT level 0 with prediction level 0 (8.86 um) in the z view");
    await page.screenshot({ path: path.join(outDir, "viewer_ct_m7L2.png") });
    const tiles = [
      { name: "z view 1:1", axis: 0, s: 10400.5 * P, u0: 7040 * P, v0: 5200 * P, du: P, w: 256, h: 256, levels: [0, 0] },
      { name: "x view 3.1 um/px", axis: 2, s: 7200.5 * P, u0: 5200 * P, v0: 10300 * P, du: 3.1, w: 192, h: 192, levels: [0, 0] },
      { name: "y view, CT level 1", axis: 1, s: 5300.25 * P, u0: 7000 * P, v0: 10200 * P, du: 2 * P, w: 160, h: 160, levels: [1, 0] },
    ];
    for (const t of tiles) {
      const b = await page.evaluate((t) => window.vcViewer.renderTile(t), t);
      const p = refTile(t, ls);
      ok(b.sha256 === p.sha256 && b.coloured > 100, `CT+m7 L2 tile (${t.name}): browser ${b.sha256.slice(0, 16)} python ${p.sha256.slice(0, 16)}, ${b.coloured} overlay pixels`);
    }
    await page.close();
  }

  // ---- 3: the prediction alone
  {
    const page = await newPage(browser, hashOf([PR2], extra));
    const ls = await state(page);
    ok(ls.length === 1 && ls[0].p0[0] === 8.86, `standalone m7 L2: one layer at ${ls[0] && ls[0].p0[0]} um`);
    const r = await settle(page);
    console.log("[3] settled after", r.s, "s:", r.st);
    await zoomZ(page, 1 / 8.86, cur[2], cur[1]);
    await settle(page);
    await page.screenshot({ path: path.join(outDir, "viewer_standalone_pred.png") });
    const t = { axis: 0, s: 10400.5 * P, u0: 7040 * P, v0: 5200 * P, du: 8.86 / 1.5, w: 160, h: 160, levels: [0] };
    const b = await page.evaluate((t) => window.vcViewer.renderTile(t), t);
    const p = refTile(t, ls);
    ok(b.sha256 === p.sha256 && b.coloured > 100, `standalone tile: browser ${b.sha256.slice(0, 16)} python ${p.sha256.slice(0, 16)}, ${b.coloured} coloured`);
    await page.close();
  }
  // 3b: a hi-dpi page with a maximised view (composited at reduced resolution)
  {
    const page = await newPage(browser, hashOf([CT2, PR2], extra), 2);
    await page.keyboard.press("f");
    await zoomZ(page, 2 / P, cur[2], cur[1]);
    await settle(page, 60);
    const info = await page.evaluate(() => { const V = window.vcViewer.S.views[0]; return { w: V.canvas.width, h: V.canvas.height, ms: window.vcViewer.stats.renderMs }; });
    ok(info.w * info.h > 1.5e6, `dpr 2, maximised z view: ${info.w}x${info.h} device px composited at reduced resolution, render ${info.ms.toFixed(0)} ms`);
    await page.screenshot({ path: path.join(outDir, "viewer_hidpi_max.png") });
    await page.close();
  }

  // ---- 4 + 5: two overlays, then the hash round trip
  {
    const hash = "#l=" + encodeURIComponent(CT2) + "&l=" + encodeURIComponent(PR2 + "*lut:red,op:0.8,bl:add") +
      "&l=" + encodeURIComponent(TH2 + "*lut:categorical,hue:green,op:0.5,bl:normal") + extra;
    const page = await newPage(browser, hash);
    const ls = await state(page);
    ok(ls.length === 3 && /k:mask/.test(ls[2].opts) && /ip:auto/.test(ls[2].opts), "three layers; the th0.2 store is a mask (nearest)");
    await zoomZ(page, 1 / P, cur[2], cur[1]);
    const r = await settle(page);
    console.log("[4] settled after", r.s, "s:", r.st);
    await page.screenshot({ path: path.join(outDir, "viewer_two_overlays.png") });
    for (const t of [
      { name: "z view 1:1", axis: 0, s: 10400.5 * P, u0: 7040 * P, v0: 5200 * P, du: P, w: 256, h: 256, levels: [0, 0, 0] },
      { name: "x view 3.1 um/px", axis: 2, s: 7200.5 * P, u0: 5200 * P, v0: 10300 * P, du: 3.1, w: 192, h: 192, levels: [0, 0, 0] },
    ]) {
      const b = await page.evaluate((t) => window.vcViewer.renderTile(t), t);
      const p = refTile(t, ls);
      ok(b.sha256 === p.sha256 && b.coloured > 100, `two overlays tile (${t.name}): browser ${b.sha256.slice(0, 16)} python ${p.sha256.slice(0, 16)}`);
    }
    // 5: change settings through the panel, then reopen the page from its hash
    await page.click("#layer-list .lrow:nth-child(2) .nm"); // the prob layer (list is top first)
    await page.selectOption("#ls-lut", "fire");
    await page.selectOption("#ls-blend", "max");
    await page.fill("#ls-thr", "100"); await page.dispatchEvent("#ls-thr", "change");
    await page.check("#ls-thr-on");
    await page.selectOption("#ls-level", "1");
    await page.click("#layer-list .lrow:nth-child(1) button[data-a=down]"); // the mask below the probabilities
    await page.click("#layer-list .lrow:nth-child(3) input[type=checkbox]"); // hide the CT
    await page.waitForTimeout(800);
    const h1 = await page.evaluate(() => location.hash);
    const s1 = await state(page);
    const v1 = await page.evaluate(() => ({ c: window.vcViewer.S.cursor, z: window.vcViewer.S.views.map((V) => V.zoom), sel: window.vcViewer.S.sel }));
    const p2 = await newPage(browser, h1);
    const s2 = await state(p2);
    const v2 = await p2.evaluate(() => ({ c: window.vcViewer.S.cursor, z: window.vcViewer.S.views.map((V) => V.zoom), sel: window.vcViewer.S.sel }));
    const same = JSON.stringify(s1.map((l) => [l.url, l.opts])) === JSON.stringify(s2.map((l) => [l.url, l.opts]));
    ok(same, "hash round trip keeps the layers, their order and settings: " + s2.map((l) => l.url.split("/").pop().slice(0, 28) + "{" + l.opts + "}").join(" / "));
    ok(v1.c.every((x, a) => Math.abs(x - v2.c[a]) < 1e-2) && v1.z.every((x, a) => Math.abs(x / v2.z[a] - 1) < 1e-4) && v1.sel === v2.sel,
      "hash round trip keeps the cursor, the zooms and the selected layer");
    const pr = s2.find((l) => l.url === PR2) || { opts: "" }, ct = s2.find((l) => l.url === CT2) || { opts: "" };
    ok(s2.map((l) => l.url).join() === [CT2, TH2, PR2].join(), "the reorder survived (CT, mask, probabilities)");
    ok(/lut:fire/.test(pr.opts) && /bl:max/.test(pr.opts) && /t:100/.test(pr.opts) && /lvl:1/.test(pr.opts) && /h:1/.test(ct.opts),
      "the panel edits reached the hash (fire, max, threshold 100, level 1, CT hidden)");
    await settle(p2, 30);
    await p2.screenshot({ path: path.join(outDir, "viewer_hash_roundtrip.png") });
    await p2.close();
    await page.close();
  }

  // ---- 6 + 7: pickers and the add-layer panel, then mouse and keys
  {
    const page = await newPage(browser, "");
    await page.waitForFunction(() => document.querySelectorAll("#sample option").length > 30, null, { timeout: 30000 });
    const nSamples = await page.evaluate(() => document.querySelectorAll("#sample option").length - 1);
    await page.selectOption("#sample", "PHerc0343P");
    await page.waitForFunction(() => window.vcViewer.S.layers.length === 1, null, { timeout: 30000 });
    const vols = await page.evaluate(() => [...document.querySelectorAll("#volume option")].map((o) => o.textContent));
    ok(nSamples > 30 && vols.length === 3, `sample list (${nSamples}) and volume list: ${vols.slice(1).join(" | ")}`);
    await page.selectOption("#volume", CT2);
    await page.waitForFunction((u) => window.vcViewer.S.layers.length === 1 && window.vcViewer.S.layers[0].st.url === u, CT2, { timeout: 30000 });
    await page.click("#add-layer");
    await page.waitForFunction(() => document.querySelectorAll("#add-list tr.ok").length >= 5, null, { timeout: 60000 });
    const rows = await page.evaluate(() => [...document.querySelectorAll("#add-list tbody tr")].map((tr) => [...tr.children].map((td) => td.textContent).join(" | ")));
    console.log("[6] add-layer panel:\n  " + rows.join("\n  "));
    ok(rows.length === 5 && rows.filter((r) => /CT L2/.test(r)).length === 2 && rows.some((r) => /mask/.test(r)), "add-layer panel lists the 2 volumes and the 3 prediction stores with kind, level, size and pitch");
    await page.screenshot({ path: path.join(outDir, "viewer_add_panel.png") });
    await page.click(`#add-list tr[data-url="${PR2}"]`);
    await page.waitForFunction(() => window.vcViewer.S.layers.length === 2, null, { timeout: 30000 });
    await page.click("#add-close");
    const ls = await state(page);
    ok(ls[1].url === PR2, "a row click adds that store on top");
    await settle(page, 60);
    // 7: mouse and keys on the z view
    const box = await page.evaluate(() => { const r = document.getElementById("view0").getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; });
    const cx = box[0] + box[2] / 2, cy = box[1] + box[3] / 2;
    const snap = () => page.evaluate(() => { const S = window.vcViewer.S, V = S.views[0], L = S.layers[S.sel];
      return { z: S.cursor[0], y: S.cursor[1], zoom: V.zoom, c: V.center.slice(), win: L.win, level: L.level, max: S.maximized, smooth: S.smooth.on, sel: S.sel, vis: S.layers.map((l) => l.visible) }; });
    const a0 = await snap();
    await page.mouse.move(cx, cy);
    await page.mouse.wheel(0, 100); await page.waitForTimeout(200);
    const a1 = await snap();
    await page.keyboard.down("Control"); await page.mouse.wheel(0, -300); await page.keyboard.up("Control"); await page.waitForTimeout(200);
    const a2 = await snap();
    await page.mouse.down(); await page.mouse.move(cx + 80, cy + 40, { steps: 5 }); await page.mouse.up(); await page.waitForTimeout(200);
    const a3 = await snap();
    await page.mouse.click(cx - 50, cy - 50); await page.waitForTimeout(200);
    const a4 = await snap();
    await page.mouse.move(cx, cy); await page.mouse.down({ button: "right" }); await page.mouse.move(cx - 60, cy, { steps: 4 }); await page.mouse.up({ button: "right" });
    const a5 = await snap();
    await page.keyboard.press(",");
    const a5b = await snap();
    await page.keyboard.press("f"); await page.keyboard.press("d"); await page.keyboard.press("n"); await page.keyboard.press("v");
    await page.waitForTimeout(300);
    const a6 = await snap();
    await page.keyboard.press("v"); await page.keyboard.press("Escape"); await page.keyboard.press("d");
    ok(a1.z !== a0.z, `wheel steps the z slice (${a0.z.toFixed(2)} -> ${a1.z.toFixed(2)} um)`);
    ok(a2.zoom > a0.zoom, `ctrl+wheel zooms (${a0.zoom.toFixed(4)} -> ${a2.zoom.toFixed(4)})`);
    ok(a3.c[0] < a2.c[0] && a3.c[1] < a2.c[1], "drag pans");
    ok(a4.y !== a3.y && a4.z === a3.z, "click moves the cursor in-plane");
    ok(a5.win < a4.win, `right-drag changes the selected layer's window (${a4.win} -> ${a5.win})`);
    ok(a6.max === 0 && a6.smooth && a6.sel !== a5.sel && a6.vis.includes(false), `keys: maximise ${a6.max}, deblock ${a6.smooth}, next layer ${a5.sel}->${a6.sel}, hide ${a6.vis}`);
    ok(a5.level === "auto" && a5b.level !== "auto", `',' fixes the selected layer's level (${a5.level} -> ${a5b.level})`);
    await page.close();
  }

  // ---- 8: a region set (volcomp + zstd inner codecs)
  {
    const rc = REG + "/region_10240_10240_10240.zarr";
    const page = await newPage(browser, hashOf([CTP4, REG], `&cu=${[10752.5 * 2.4, 10752.5 * 2.4, 10752.5 * 2.4].join(",")}`));
    const b = await page.evaluate((u) => window.vcViewer.chunkChecksum(u, 1, 1, 1), rc);
    const p = py([PYC, "url", rc, "1", "1", "1"]);
    ok(b.zstd && b.sha256 === p.plain, `region chunk (1,1,1), zstd -> volcomp: browser ${String(b.sha256).slice(0, 16)} python ${String(p.plain).slice(0, 16)}`);
    const ls = await state(page);
    ok(ls.length === 2 && ls[1].p0[0] === 2.4 && ls[0].p0[0] === 2.4, `region set layer at ${ls[1] && ls[1].p0[0]} um over the ${ls[0] && ls[0].p0[0]} um CT (${ls[1] && ls[1].rule})`);
    await zoomZ(page, 1 / 2.4, 10752 * 2.4, 10752 * 2.4);
    const r = await settle(page);
    console.log("[8] settled after", r.s, "s:", r.st);
    const t = { axis: 0, s: 10752.5 * 2.4, u0: 10700 * 2.4, v0: 10700 * 2.4, du: 2.4, w: 128, h: 128, levels: [0, 0] };
    const tb = await page.evaluate((t) => window.vcViewer.renderTile(t), t);
    ok(tb.coloured > 100, `region set renders over the CT (${tb.coloured} coloured pixels in a 128x128 tile)`);
    await page.screenshot({ path: path.join(outDir, "viewer_regions.png") });
    await page.close();
  }

  // ---- 9: a settled view stays settled (no fetches, decodes or redraws for 10 s), with three
  // layers whose visible sets together exceed the cache (the chunks on screen are never evicted)
  const counters = (page) => page.evaluate(() => ({ req: window.vcViewer.stats.requests, dec: window.vcViewer.stats.decoded, ren: window.vcViewer.stats.renders }));
  async function idle(page, name) {
    const a = await counters(page);
    await page.waitForTimeout(10000);
    const b = await counters(page);
    ok(b.req === a.req && b.dec === a.dec && b.ren === a.ren,
      `${name}: 10 s idle after settling: ${b.req - a.req} requests, ${b.dec - a.dec} decodes, ${b.ren - a.ren} redraws`);
  }
  const rss = () => { // resident memory of the browser's renderer processes (MB)
    const ps = execFileSync("ps", ["-eo", "rss,args"], { encoding: "utf8" }).split("\n").filter((l) => /--type=renderer/.test(l) && /headless|chrom/i.test(l));
    const r = ps.map((l) => +l.trim().split(/\s+/)[0] / 1024);
    return { max: Math.max(0, ...r).toFixed(0), n: r.length };
  };
  {
    const page = await newPage(browser, hashOf([CTP4, P4PR, P4TH]), 1.25, 1920, 1080);
    const r = await settle(page, 120);
    console.log("[9] Paris4 3 layers (default view, 1920x1080 @1.25) settled after", r.s, "s:", r.st);
    await page.screenshot({ path: path.join(outDir, "viewer_paris4_3layers.png") });
    await idle(page, "Paris4 CT + m7 L2 prob + th0.2 mask, default view, default cache");
    const m0 = rss();
    // browse: 1:1 zoom in the z view, then 20 slices (fills the cache with visible + prefetched chunks)
    await page.evaluate(() => { const V = window.vcViewer.S.views[0], S = window.vcViewer.S; V.zoom = 1 / 2.4; V.center = [S.cursor[2], S.cursor[1]]; window.vcViewer.requestRender(); });
    await settle(page, 120);
    const box = await page.evaluate(() => { const r = document.getElementById("view0").getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; });
    await page.mouse.move(box[0], box[1]);
    for (let i = 0; i < 20; i++) { await page.mouse.wheel(0, 100); await page.waitForTimeout(150); }
    await page.mouse.move(0, 0);
    const r2 = await settle(page, 120);
    console.log("[9] after zoom 1:1 and 20 slices:", r2.st);
    await idle(page, "Paris4 3 layers after browsing");
    const m1 = rss();
    const cs = await page.evaluate(() => ({ b: window.vcViewer.cache.bytes, cap: window.vcViewer.cache.cap }));
    console.log(`[9] memory at 3 layers, ${(cs.cap / 2 ** 30).toFixed(1)} GB cap: decoded cache ${(cs.b / 2 ** 20).toFixed(0)} MB; renderer RSS ${m0.max} MB settled, ${m1.max} MB after browsing (${m1.n} renderer process(es))`);
    await page.close();
  }
  {
    const page = await newPage(browser, hashOf([CT2, PR2, TH2], extra), 1.25, 1920, 1080);
    await page.selectOption("#cache-cap", "1024");
    await page.evaluate(() => { for (const V of window.vcViewer.S.views) V.zoom = 0.3; const S = window.vcViewer.S;
      S.views[0].center = [S.cursor[2], S.cursor[1]]; S.views[1].center = [S.cursor[2], S.cursor[0]]; S.views[2].center = [S.cursor[1], S.cursor[0]]; window.vcViewer.requestRender(); });
    const r = await settle(page, 120);
    console.log("[9] PHerc0343P 3 layers at 0.3 px/um, 1 GB cache, settled after", r.s, "s:", r.st);
    const warn = await page.evaluate(() => document.getElementById("status-warn").textContent);
    ok(/cache share/.test(warn), "a layer whose visible set exceeds its share says so: " + warn);
    await idle(page, "3 layers at 0.3 px/um with a 1 GB cache (visible sets > shares)");
    await page.screenshot({ path: path.join(outDir, "viewer_small_cache.png") });
    // zarr v2 (the published blosc store) is rejected with a clear message, once
    const ly = await page.evaluate((u) => window.vcViewer.addLayer(u).then((l) => !!l), S3TH);
    const msg = await page.evaluate(() => document.getElementById("status-warn").textContent);
    ok(!ly && /zarr v2/.test(msg) && /volcomp copy/.test(msg), "zarr v2 blosc store rejected: " + msg);
    await idle(page, "after the rejected v2 store");
    await page.close();
  }

  await browser.close();
  console.log(`${fails ? fails + " FAIL(s)" : "all PASS"} in ${((Date.now() - t00) / 1000).toFixed(0)} s; screenshots in ${path.resolve(outDir)}`);
  process.exit(fails ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
