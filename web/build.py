"""Inline src/worker.js, src/main.js and the wasm (base64) into src/index.html."""
import base64, pathlib, sys

here = pathlib.Path(__file__).parent
wasm_path, out_path = (pathlib.Path(a) for a in sys.argv[1:3])
html = (here / "src/index.html").read_text()
worker = (here / "src/worker.js").read_text()
main = (here / "src/main.js").read_text()
for name, text in (("worker.js", worker), ("main.js", main)):
    if "</script" in text.lower():
        sys.exit(f"{name} contains '</script' and cannot be inlined")
b64 = base64.b64encode(wasm_path.read_bytes()).decode()
b64 = "\n".join(b64[i:i + 120] for i in range(0, len(b64), 120))
for token, text in (("/*@WORKER@*/", worker), ("/*@WASM_B64@*/", b64), ("/*@MAIN@*/", main)):
    if html.count(token) != 1:
        sys.exit(f"index.html must contain {token} exactly once")
    html = html.replace(token, text)
out_path.write_text(html)
print(f"{out_path}: {out_path.stat().st_size:,} bytes (wasm {wasm_path.stat().st_size:,})")
