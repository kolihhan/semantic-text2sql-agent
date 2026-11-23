#!/usr/bin/env bash
set -euo pipefail
test -n "$DB_ZIP_SHA"
mkdir -p /tmp/bird/unpacked /tmp/bird/databases
curl -fL --retry 3 https://huggingface.co/datasets/HAL-9001/bird-dev/resolve/main/dev.zip -o /tmp/bird/dev.zip
echo "$DB_ZIP_SHA  /tmp/bird/dev.zip" | sha256sum -c -
unzip -q /tmp/bird/dev.zip -d /tmp/bird/unpacked
INNER=$(find /tmp/bird/unpacked -type f -name dev_databases.zip ! -path '*/__MACOSX/*' | head -n1)
test -n "$INNER"
unzip -q "$INNER" -d /tmp/bird/databases
DB_ROOT=$(find /tmp/bird/databases -type d -name dev_databases ! -path '*/__MACOSX/*' | head -n1)
if [ -z "$DB_ROOT" ]; then DB_ROOT=/tmp/bird/databases; fi
test -d "$DB_ROOT"
echo "DB_ROOT=$DB_ROOT" >> "$GITHUB_ENV"
echo "BIRD pinned physical database restored"
