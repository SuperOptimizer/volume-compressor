#!/bin/bash
# VM bootstrap for the volcomp export fleet. Run as root with the environment
# exported by fleet.py bootstrap: ROLE, COMMIT, REPO, COORDINATOR, SFTP,
# NETRC_CONTENT, Q, PARALLEL, SAMPLES, PORT. Idempotent.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# fresh images run unattended-upgrades at boot; wait for the package locks
for i in $(seq 1 120); do
  fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock >/dev/null 2>&1 || break
  sleep 5
done

# latest stable clang from apt.llvm.org (the distro clang is too old for C23)
if ! ls /usr/bin/clang-[0-9]* >/dev/null 2>&1 || [ "$(ls /usr/bin/clang-[0-9]* | sed 's/.*clang-//' | sort -n | tail -1)" -lt 18 ]; then
  apt-get update -qq
  apt-get install -y -qq git curl python3 wget lsb-release software-properties-common gnupg sshpass openssh-client >/dev/null
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
command -v sshpass >/dev/null || apt-get install -y -qq sshpass >/dev/null
echo "using $($CLANG --version | head -1), $(cmake --version | head -1)"
grep -q -w avx2 /proc/cpuinfo && grep -q -w fma /proc/cpuinfo || { echo "CPU lacks AVX2/FMA"; exit 1; }

# source + build at the pinned commit
if [ ! -d /opt/volume-compressor/.git ]; then
  git clone -q "$REPO" /opt/volume-compressor
fi
cd /opt/volume-compressor
git fetch -q origin
git checkout -q --detach "origin/$COMMIT" 2>/dev/null || git checkout -q --detach "$COMMIT"
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CLANG" \
  -DVOLCOMP_NATIVE=ON -DVOLCOMP_BENCH=OFF >/dev/null
cmake --build build/release --target volcomp_cli >/dev/null
install -m 755 build/release/volcomp /usr/local/bin/volcomp
volcomp encode 2>&1 | grep -q usage || test -x /usr/local/bin/volcomp
install -m 755 tools/export/worker.py /usr/local/bin/volcomp-worker
install -m 755 tools/export/coordinator.py /usr/local/bin/volcomp-coordinator

umask 077
if [ -n "${NETRC_CONTENT:-}" ]; then
  printf '%s\n' "$NETRC_CONTENT" > /etc/volcomp-netrc
  printf '%s\n' "$NETRC_CONTENT" > /root/.netrc   # lftp / curl on the VM itself
fi
umask 022

if [ "$ROLE" = "deleter" ]; then
  command -v lftp >/dev/null || apt-get install -y -qq lftp >/dev/null
  [ -n "${DELETE_PATHS:-}" ] || { echo "deleter needs DELETE_PATHS"; exit 1; }
  HOST=$(sed -n 's|sftp://\([^/:]*\).*|\1|p' <<<"$SFTP"); PORT=$(sed -n 's|sftp://[^/:]*:\([0-9]*\).*|\1|p' <<<"$SFTP")
  LOGIN=$(awk '{for(i=1;i<NF;i++) if($i=="login") print $(i+1)}' /etc/volcomp-netrc | head -1)
  cat > /usr/local/bin/volcomp-delete <<EOF2
#!/bin/bash
# recursive delete of the listed remote trees over one persistent SFTP session; logs to /var/log/volcomp-delete.log
export LFTP_PASSWORD=\$(awk '{for(i=1;i<NF;i++) if(\$i=="password") print \$(i+1)}' /etc/volcomp-netrc | head -1)
exec lftp --env-password -u $LOGIN -p ${PORT:-22} "sftp://$HOST" -e "set sftp:auto-confirm yes; set sftp:connect-program 'ssh -a -x -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'; set net:timeout 60; set net:max-retries 20; set net:reconnect-interval-base 5; set cmd:fail-exit no; $(for d in $DELETE_PATHS; do printf 'echo == rm -r %s; rm -r -f %s; ' "$d" "$d"; done) echo == remaining:; ls /; bye"
EOF2
  chmod 755 /usr/local/bin/volcomp-delete
  cat > /etc/systemd/system/volcomp-delete.service <<'EOF2'
[Unit]
Description=volcomp: delete old SFTP export trees
After=network-online.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c '/usr/local/bin/volcomp-delete >> /var/log/volcomp-delete.log 2>&1'
RemainAfterExit=yes
EOF2
  systemctl daemon-reload
  systemctl start --no-block volcomp-delete
  echo "deleter started: DELETE_PATHS=$DELETE_PATHS (log: /var/log/volcomp-delete.log)"
  exit 0
fi

if [ "$ROLE" = "worker" ]; then
  umask 077
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
