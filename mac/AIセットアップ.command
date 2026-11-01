#!/bin/zsh
# 声とノイズの分離 — Mac用 AIフルセットアップ
# AIエンジン（MossFormer2 / DeepFilterNet3）込みで全部入れるための一括スクリプト。
# ダブルクリックで実行。途中失敗しても最後まで進み、結果を表示します。

cd "$(dirname "$0")/.." || exit 1
echo "=== 声とノイズの分離: AIセットアップ開始 ==="
echo "作業フォルダ: $(pwd)"
echo

# 1) 仮想環境
if [ ! -d .venv ]; then
  echo "--- .venv を作成します ---"
  python3 -m venv .venv
else
  echo "--- .venv は既にあります（そのまま使います）---"
fi
PIP=".venv/bin/pip"
PY=".venv/bin/python"
"$PIP" install -q --upgrade pip
echo

# 2) 基本ライブラリ
echo "--- 基本ライブラリをインストールします（fastapi / uvicorn など）---"
"$PIP" install fastapi uvicorn python-multipart numpy scipy
echo

# 3) PyTorch（Mac通常版 = Apple Silicon の MPS 対応ビルド。そのまま入れる）
echo "--- PyTorch をインストールします（少し時間がかかります）---"
"$PIP" install torch torchaudio
echo

# 4) MossFormer2（ClearVoice）
echo "--- ClearVoice（MossFormer2用）をインストールします ---"
"$PIP" install clearvoice soundfile
echo

# 4.5) RE-USE (NVIDIA NemotronLabs RE-USE) — モデル一式を checkpoints/ へ
echo "--- RE-USE モデルをダウンロードします（約38MB）---"
"$PIP" install safetensors huggingface_hub
if "$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/NVIDIA-NemotronLabs-RE-USE",
                  local_dir="checkpoints/NVIDIA-NemotronLabs-RE-USE")
print("RE-USE snapshot downloaded")
PYEOF
then
  echo "✅ RE-USE モデルのダウンロード完了"
else
  echo "⚠️ RE-USE のダウンロードに失敗しました（ネット接続を確認してください）。他のエンジンは使えます。"
fi
echo

# 5) DeepFilterNet3（Rustビルドが必要な環境では失敗することがある）
echo "--- DeepFilterNet をインストールします ---"
if ! "$PIP" install deepfilternet; then
  echo "⚠️ DeepFilterNet のインストールに失敗しました。"
  echo "   DeepFilterNet は Rust が必要な場合があります（https://rustup.rs で Rust を入れてから再実行してください）。"
  echo "   MossFormer2 とスペクトラル方式は使えるので、このまま続行します。"
fi
echo

# 6) 動作確認
echo "--- 動作確認 ---"
"$PY" - <<'PYEOF'
import sys

ok = []

try:
    import torch
    mps = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    print(f"✅ torch {torch.__version__}（MPS利用可: {mps}）")
    ok.append("torch")
except Exception as e:
    print(f"❌ torch の読み込みに失敗: {e}")

try:
    import clearvoice  # noqa: F401
    print("✅ clearvoice（MossFormer2）読み込みOK")
    ok.append("clearvoice")
except Exception as e:
    print(f"❌ clearvoice の読み込みに失敗: {e}")

try:
    sys.path.insert(0, ".")
    from core import reuse_engine
    if reuse_engine.available():
        print("✅ RE-USE（NemotronLabs）モデル配置済み")
        ok.append("reuse")
    else:
        print("⚠️ RE-USE は未配置です（上のダウンロード結果を確認してください）")
except Exception as e:
    print(f"⚠️ RE-USE の確認に失敗: {e}")

try:
    sys.path.insert(0, ".")
    from core.separator import _patch_torchaudio_for_dfn
    _patch_torchaudio_for_dfn()
    import df  # noqa: F401
    print("✅ DeepFilterNet（df）読み込みOK")
    ok.append("df")
except Exception as e:
    print(f"⚠️ DeepFilterNet（df）は未使用状態です: {e}")

print("確認できたもの:", ", ".join(ok) if ok else "なし")
PYEOF
echo

# 7) モデルの先行ダウンロード（MossFormer2_SE_48K を1回ロードする）
echo "--- MossFormer2_SE_48K モデルを先にダウンロードします（数分かかることがあります）---"
if "$PY" - <<'PYEOF'
from clearvoice import ClearVoice
cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
print("モデル読み込み完了")
PYEOF
then
  echo "✅ MossFormer2_SE_48K のダウンロード完了"
else
  echo "⚠️ モデルの先行ダウンロードに失敗しました（ネット接続を確認してください）。"
  echo "   初回利用時にも自動でダウンロードされるので、このまま進んでも大丈夫です。"
fi
echo

echo "=== セットアップ完了 ==="
echo "「一発起動.command」をダブルクリックするとアプリが Safari で開きます。"
echo "このウィンドウは閉じてOKです。"
read -r "?Enterキーで閉じます: "
