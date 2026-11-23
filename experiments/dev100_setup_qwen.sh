#!/usr/bin/env bash
set -euo pipefail
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | sudo tar --zstd -x -C /usr
ollama serve > /tmp/ollama.log 2>&1 &
for i in {1..30}; do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then break; fi
  sleep 1
done
ollama pull "$MODEL"
python - <<'PY'
import json,os,urllib.request
models=json.load(urllib.request.urlopen("http://127.0.0.1:11434/api/tags"))["models"]
matched=[row["digest"] for row in models if row.get("name")==os.environ["MODEL"] or row.get("model")==os.environ["MODEL"]]
assert matched==[os.environ["MODEL_DIGEST"]], matched
print("EXACT_FIXED_QWEN9B_DIGEST_VERIFIED")
PY
