#!/bin/zsh
# 声とノイズの分離 — Macアプリ版ランチャー
# Web版と同じサーバー(app.py)をローカルで起動し、Safariで開く。
# （Kantaの運用ルール: ローカルツールはChromeでなくSafariで開く）
cd "$(dirname "$0")/.."
PORT=8200
URL="http://localhost:${PORT}"

if curl -s -o /dev/null --max-time 2 "$URL"; then
  open -a Safari "$URL"
  exit 0
fi

# 初回だけ venv を作って依存を入れる
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q fastapi uvicorn python-multipart numpy scipy
fi

nohup .venv/bin/python app.py > separator.log 2>&1 &
for i in {1..30}; do
  if curl -s -o /dev/null --max-time 2 "$URL"; then break; fi
  sleep 1
done
open -a Safari "$URL"
