#!/usr/bin/env bash
# Build an agentfork guest rootfs from a Firecracker CI base squashfs.
#
# Bakes in, so a forked guest is ready the moment it resumes:
#   - the guest exec agent (agentfork/sandbox/guest_agent.py) as a systemd
#     service on vsock port 52;
#   - static network config matching agentfork/sandbox/netns.py (guest
#     172.16.0.2/30 via 172.16.0.1 on eth0) plus a nameserver, so a guest
#     that gets a netns has working DNS + egress with no DHCP;
#   - a boot-time identity regen unit: fresh machine-id and SSH host keys on
#     first boot after a restore, so snapshot clones aren't identical hosts.
#
# Usage:
#   tools/build_rootfs.sh --base ubuntu.squashfs --out rootfs.squashfs \
#       [--nameserver 1.1.1.1]
#
# Requires root (squashfs-tools). Runs on Linux; the agent needs python3 in
# the base image (the Firecracker CI Ubuntu images have it).
set -euo pipefail

NAMESERVER=1.1.1.1
BASE="" OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --nameserver) NAMESERVER="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$BASE" ] && [ -n "$OUT" ] || { echo "need --base and --out" >&2; exit 2; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
# unsquashfs runs under sudo and writes root-owned files, so cleanup needs it
trap 'sudo rm -rf "$WORK"' EXIT
ROOT="$WORK/squashfs-root"

sudo unsquashfs -q -d "$ROOT" "$BASE" >/dev/null
[ -x "$ROOT/usr/bin/python3" ] || { echo "base image lacks python3" >&2; exit 1; }

# 1. exec agent + service
sudo install -D -m 0755 "$REPO/agentfork/sandbox/guest_agent.py" \
  "$ROOT/usr/local/bin/guest_agent.py"
sudo tee "$ROOT/etc/systemd/system/guest-agent.service" >/dev/null <<'UNIT'
[Unit]
Description=agentfork guest exec agent
After=network-online.target
[Service]
ExecStart=/usr/bin/python3 /usr/local/bin/guest_agent.py 52
Restart=always
[Install]
WantedBy=multi-user.target
UNIT

# 2. static networking matching netns.py (systemd-networkd)
sudo tee "$ROOT/etc/systemd/network/10-eth0.network" >/dev/null <<'NET'
[Match]
Name=eth0
[Network]
Address=172.16.0.2/30
Gateway=172.16.0.1
NET
echo "nameserver $NAMESERVER" | sudo tee "$ROOT/etc/resolv.conf" >/dev/null

# 3. identity regen on boot: a restored clone must not keep the parent's
#    machine-id / host keys (they'd collide across siblings)
sudo tee "$ROOT/etc/systemd/system/regen-identity.service" >/dev/null <<'UNIT'
[Unit]
Description=agentfork per-clone identity regeneration
Before=guest-agent.service sshd.service
ConditionPathExists=!/var/lib/agentfork-identity-done
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'rm -f /etc/machine-id && systemd-machine-id-setup && rm -f /etc/ssh/ssh_host_* && (ssh-keygen -A || true) && touch /var/lib/agentfork-identity-done'
[Install]
WantedBy=multi-user.target
UNIT

# 4. enable everything (symlink into the multi-user target)
for unit in guest-agent regen-identity systemd-networkd; do
  sudo ln -sf "/etc/systemd/system/${unit}.service" \
    "$ROOT/etc/systemd/system/multi-user.target.wants/${unit}.service" \
    2>/dev/null || true
done
sudo ln -sf /lib/systemd/system/systemd-networkd.service \
  "$ROOT/etc/systemd/system/multi-user.target.wants/systemd-networkd.service" \
  2>/dev/null || true

sudo mksquashfs "$ROOT" "$OUT" -quiet -noappend
echo "built $OUT"
