#!/bin/bash
# VM bootstrap for the volcomp export fleet. Run as root with the environment
# exported by fleet.py bootstrap: ROLE, COMMIT, REPO, COORDINATOR, SFTP,
# NETRC_CONTENT, Q, PARALLEL, SAMPLES, PORT. Idempotent.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# latest stable clang from apt.llvm.org (the distro clang is too old for C23)
if ! ls /usr/bin/clang-[0-9]* >/dev/null 2>&1 || [ "$(ls /usr/bin/clang-[0-9]* | sed 's/.*clang-//' | sort -n | tail -1)" -lt 18 ]; then
  apt-get update -qq
  apt-get install -y -qq git curl python3 wget lsb-release software-properties-common gnupg >/dev/null
  curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh
  bash /tmp/llvm.sh >/dev/null 2>&1 || bash /tmp/llvm.sh   # no version argument = current stable
fi
CLANG=$(ls /usr/bin/clang-[0-9]* | sort -t- -k2 -n | tail -1)
ln -sf "$CLANG" /usr/local/bin/clang
# latest CMake release (official binary tarball) + ninja
if ! [ -x /opt/cmake/bin/cmake ]; then
  apt-get install -y -qq ninja-build curl >/dev/null
  CMAKE_URL=$(curl -fsSL -H 'User-Agent: volcomp-bootstrap' https://api.github.com/repos/Kitware/CMake/releases/latest \
    | grep -o 'https://[^"]*cmake-[0-9.]*-linux-x86_64.tar.gz' | head -1)
  curl -fsSL "$CMAKE_URL" -o /tmp/cmake.tgz
  rm -rf /opt/cmake && mkdir -p /opt/cmake && tar -xzf /tmp/cmake.tgz -C /opt/cmake --strip-components=1
fi
ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake
echo "using $($CLANG --version | head -1), $(cmake --version | head -1)"
grep -q -w avx2 /proc/cpuinfo && grep -q -w fma /proc/cpuinfo || { echo "CPU lacks AVX2/FMA"; exit 1; }

# source + build at the pinned commit
if [ ! -d /opt/volume-compressor/.git ]; then
  git clone -q "$REPO" /opt/volume-compressor
fi
cd /opt/volume-compressor
git fetch -q origin
git checkout -q "$COMMIT" 2>/dev/null || git checkout -q "origin/$COMMIT"
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CLANG" \
  -DVOLCOMP_NATIVE=ON -DVOLCOMP_BENCH=OFF >/dev/null
cmake --build build/release --target volcomp_cli >/dev/null
install -m 755 build/release/volcomp /usr/local/bin/volcomp
volcomp 2>&1 | grep -q usage
install -m 755 tools/export/worker.py /usr/local/bin/volcomp-worker
install -m 755 tools/export/coordinator.py /usr/local/bin/volcomp-coordinator

if [ "$ROLE" = "worker" ]; then
  umask 077
  if [ -n "${NETRC_CONTENT:-}" ]; then printf '%s\n' "$NETRC_CONTENT" > /etc/volcomp-netrc; fi
  cat > /etc/volcomp-worker.env <<EOF
COORDINATOR=$COORDINATOR
SFTP=$SFTP
Q=$Q
PARALLEL=$PARALLEL
SAMPLES=$SAMPLES
EOF
  cat > /etc/systemd/system/volcomp-worker.service <<'EOF'
[Unit]
Description=volcomp export worker
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile=/etc/volcomp-worker.env
ExecStart=/usr/bin/python3 /usr/local/bin/volcomp-worker run --coordinator ${COORDINATOR} --volcomp /usr/local/bin/volcomp \
  --tmp /dev/shm/volcomp --sftp ${SFTP} --netrc /etc/volcomp-netrc --q ${Q} --parallel ${PARALLEL} --samples ${SAMPLES}
Restart=always
RestartSec=10
Nice=5

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable -q volcomp-worker
  systemctl restart volcomp-worker
  sleep 2
  systemctl is-active volcomp-worker
else
  mkdir -p /var/lib/volcomp
  cat > /etc/systemd/system/volcomp-coordinator.service <<EOF
[Unit]
Description=volcomp export coordinator
After=network-online.target

[Service]
ExecStart=/usr/bin/python3 /usr/local/bin/volcomp-coordinator serve --db /var/lib/volcomp/export.db --bind 0.0.0.0 --port $PORT --lease 900
Restart=always
RestartSec=5
WorkingDirectory=/var/lib/volcomp

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable -q volcomp-coordinator
  systemctl restart volcomp-coordinator
  sleep 1
  systemctl is-active volcomp-coordinator
  echo "coordinator up; build the queue with: volcomp-coordinator manifest --db /var/lib/volcomp/export.db"
fi
