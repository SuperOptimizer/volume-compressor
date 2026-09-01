#!/usr/bin/env python3
"""Blue Lobster VM fleet for the volcomp export (Python 3 standard library only).

  fleet.py launch --role coordinator [--type v1_cpu_medium]
  fleet.py launch --role worker --count N [--type v1_cpu_medium]
  fleet.py launch --role deleter                  one small VM that deletes the old SFTP trees (bootstrap --delete-paths)
  fleet.py list                                   fleet instances (named volcomp-coordinator / volcomp-worker-NNN)
  fleet.py wait                                   block until every fleet VM has an IP and answers SSH
  fleet.py bootstrap [--role worker|coordinator] [--commit SHA]
      ssh into each VM: install clang/cmake/git/curl/python3, clone the repo at the
      given commit, build, install `volcomp`, and (workers) start the worker service
      pointed at the coordinator's internal IP. Idempotent; rerun after adding VMs.
  fleet.py ssh NAME [CMD...]                      convenience
  fleet.py destroy [--role worker|coordinator] [--yes]

Credentials: ~/bluelobster.txt (API key, one line) or $BLUELOBSTER_API_KEY; the SSH
public key in ~/.ssh/id_ed25519.pub (or --ssh-pub) is installed on every VM.
Worker settings (SFTP destination + netrc, q, parallelism) are pushed at bootstrap
from --sftp / --netrc / --q / --parallel and live in /etc/volcomp-worker.env on
the VM; edit that file and `systemctl restart volcomp-worker` to change them.
"""
import argparse
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://api.bluelobster.ai/api/v1"
TAG = {"project": "volcomp-export"}
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/SuperOptimizer/volume-compressor"


def api_key():
    k = os.environ.get("BLUELOBSTER_API_KEY")
    if not k:
        with open(os.path.expanduser("~/bluelobster.txt")) as f:
            k = f.read().strip()
    return k


def call(method, path, body=None, retries=5):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={"X-API-Key": api_key(), "Content-Type": "application/json",
                                          "User-Agent": "volcomp-fleet/1.0 (curl-compatible)"})  # CF blocks urllib UA
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise SystemExit(f"{method} {path} -> HTTP {e.code}: {msg}")
        except urllib.error.URLError as e:
            if attempt + 1 < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise SystemExit(f"{method} {path}: {e}")


def role_of(inst):
    """Fleet membership and role come from the instance name (the API does not
    persist launch metadata): volcomp-coordinator / volcomp-worker-NNN."""
    n = inst.get("name") or ""
    if n == "volcomp-coordinator":
        return "coordinator"
    if n.startswith("volcomp-worker-"):
        return "worker"
    if n == "volcomp-deleter":
        return "deleter"
    return None


def instances(role=None):
    out = []
    for i in call("GET", "/instances") or []:
        r = role_of(i)
        if r is None or (role and r != role):
            continue
        out.append(i)
    return sorted(out, key=lambda i: i.get("name") or "")


def ssh_pub(a):
    with open(os.path.expanduser(a.ssh_pub)) as f:
        return f.read().strip()


def ssh_cmd(ip, user, cmd=None, tty=False):
    base = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=15", f"{user}@{ip}"]
    if tty:
        base.insert(1, "-t")
    return base + ([cmd] if cmd else [])


# ----------------------------------------------------------------------------- commands


def cmd_launch(a):
    pub = ssh_pub(a)
    existing = {i.get("name") for i in instances()}
    n = a.count if a.role == "worker" else 1
    launched = []
    for k in range(1000):
        if len(launched) >= n:
            break
        # workers get a random suffix: the API does not list an instance until creation
        # finishes, so sequential numbering can collide with launches still in flight
        name = f"volcomp-worker-{secrets.token_hex(3)}" if a.role == "worker" else f"volcomp-{a.role}"
        if name in existing:
            continue
        body = {"region": a.region, "instance_type": a.type, "ssh_key": pub, "username": a.user, "name": name,
                "template_name": a.template, "metadata": {**TAG, "role": a.role}}
        r = call("POST", "/instances/launch-instance", body)
        print(f"launched {name}: {json.dumps(r)[:200]}")
        launched.append(name)
    if a.role != "worker" and not launched:
        print(f"{a.role} already exists")


def cmd_list(a):
    for i in instances(a.role):
        print(f"{i.get('name'):24} {role_of(i):12} {i.get('power_status', ''):10} ip={i.get('ip_address')} "
              f"internal={i.get('internal_ip')} {i.get('cpu_cores')}c/{i.get('memory')}G {i.get('instance_type')} uuid={i.get('uuid')}")


def wait_ssh(ip, user, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = subprocess.run(ssh_cmd(ip, user, "true"), capture_output=True)
        if r.returncode == 0:
            return True
        time.sleep(10)
    return False


def cmd_wait(a):
    t0 = time.time()
    while True:
        pending = [i for i in instances(a.role) if not i.get("ip_address")]
        if not pending or time.time() - t0 > a.timeout:
            break
        print(f"waiting for IPs: {[i.get('name') for i in pending]}")
        time.sleep(15)
    for i in instances(a.role):
        ok = wait_ssh(i["ip_address"], a.user, max(30, a.timeout - (time.time() - t0)))
        print(f"{i['name']}: ssh {'ok' if ok else 'TIMEOUT'} ({i['ip_address']})")


def coordinator_ip(prefer_internal=True):
    cs = instances("coordinator")
    if not cs:
        raise SystemExit("no coordinator instance; run: fleet.py launch --role coordinator")
    c = cs[0]
    return (c.get("internal_ip") if prefer_internal and c.get("internal_ip") else c.get("ip_address")), c


def cmd_bootstrap(a):
    with open(os.path.join(HERE, "bootstrap.sh")) as f:
        script = f.read()
    coord_ip, coord = coordinator_ip(not a.public_coordinator) if instances("coordinator") else (None, {"ip_address": None})
    netrc = ""
    if a.netrc:
        with open(os.path.expanduser(a.netrc)) as f:
            netrc = f.read()
    targets = instances(a.role)
    if not targets:
        raise SystemExit("no instances to bootstrap")
    def one(i):
        role = role_of(i)
        env = {"ROLE": role, "COMMIT": a.commit, "REPO": REPO, "COORDINATOR": f"http://{coord_ip}:{a.port}",
               "SFTP": a.sftp or "", "NETRC_CONTENT": netrc, "Q": str(a.q), "PARALLEL": str(a.parallel),
               "SAMPLES": str(a.samples), "PORT": str(a.port), "DELETE_PATHS": a.delete_paths or ""}
        exports = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
        try:
            r = subprocess.run(ssh_cmd(i["ip_address"], a.user, "sudo -E bash -s"), input=exports + script, text=True,
                               capture_output=not a.verbose, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            return f"== {i['name']} ({role}) at {i['ip_address']}: TIMEOUT after {a.timeout}s"
        if r.returncode:
            return f"== {i['name']} ({role}) at {i['ip_address']}: FAILED ({r.returncode}): {(r.stderr or '')[-800:]}"
        return f"== {i['name']} ({role}) at {i['ip_address']}: ok"

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for line in ex.map(one, targets):
            print(line, flush=True)
    print(f"coordinator API: http://{coord['ip_address']}:{a.port}/status (workers use {coord_ip})")


def cmd_ssh(a):
    for i in instances():
        if i.get("name") == a.name:
            os.execvp("ssh", ssh_cmd(i["ip_address"], a.user, " ".join(a.cmd) if a.cmd else None, tty=not a.cmd))
    raise SystemExit(f"no fleet instance named {a.name}")


def cmd_destroy(a):
    targets = instances(a.role)
    if not targets:
        print("nothing to destroy")
        return
    for i in targets:
        print(f"  {i['name']} {i.get('uuid')} {i.get('ip_address')}")
    if not a.yes:
        if input(f"delete these {len(targets)} instances? [y/N] ").strip().lower() != "y":
            return
    for i in targets:
        r = call("DELETE", f"/instances/{i['uuid']}")
        print(f"deleted {i['name']}: {json.dumps(r)[:120]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default="forrest", help="VM account name")
    ap.add_argument("--ssh-pub", default="~/.ssh/id_ed25519.pub")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--role", choices=["coordinator", "worker", "deleter"], required=True)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--type", default="v1_cpu_medium")
    p.add_argument("--region", default="igl")
    p.add_argument("--template", default="UBUNTU-22-04")
    p.set_defaults(fn=cmd_launch)
    p = sub.add_parser("list")
    p.add_argument("--role")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("wait")
    p.add_argument("--role")
    p.add_argument("--timeout", type=int, default=900)
    p.set_defaults(fn=cmd_wait)
    p = sub.add_parser("bootstrap")
    p.add_argument("--role")
    p.add_argument("--commit", default="main")
    p.add_argument("--sftp", help="sftp://dl.ash2txt.org:9238/volcomp")
    p.add_argument("--netrc", help="local netrc file whose contents are installed on the workers")
    p.add_argument("--q", type=float, default=8.0)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--public-coordinator", action="store_true", help="workers reach the coordinator by public IP")
    p.add_argument("--delete-paths", help="deleter role: space-separated remote dirs to remove recursively")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--jobs", type=int, default=8, help="VMs bootstrapped concurrently")
    p.add_argument("--timeout", type=int, default=1500, help="seconds per VM before giving up on it")
    p.set_defaults(fn=cmd_bootstrap)
    p = sub.add_parser("ssh")
    p.add_argument("name")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_ssh)
    p = sub.add_parser("destroy")
    p.add_argument("--role")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_destroy)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
