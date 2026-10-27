# Macアプリ版

中身は Web版と同じ `app.py`（共通コア `core/separator.py`）。配布形態だけ Mac 用にしたもの。

## 使い方（今すぐ使える形）
1. このフォルダ一式を Mac に置く
2. `mac/一発起動.command` をダブルクリック
   - 初回だけ Python venv を作って依存（fastapi / uvicorn / numpy / scipy）を入れる
   - サーバーが `localhost:8200` で立ち、Safari で画面が開く（Kanta の運用に合わせ Safari 固定）

## .app 化する場合
`一発起動.command` を Automator「アプリケーション」で包むか、
既存ツールの .app と同じ最小バンドル方式（Info.plist + launcher シム）が使える。
※実行ビット注意: files API 経由で .command を置く時は、既存の実行可能ファイルを
コピーしてから中身を編集する方式で実行ビットを維持すること（AGENTS.md の教訓）。

## AIエンジン（任意・無料・VMで検証済みの手順）
```
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install clearvoice soundfile   # MossFormer2（最高品質）
.venv/bin/pip install deepfilternet          # DeepFilterNet3（高速・自然）
```
- 数GB級。Apple Silicon では torch が MPS を使う（上記CPUホイール指定は VM/Linux 用。Macでは通常の `pip install torch torchaudio` でMPS版が入る）
- DeepFilterNet の拡張がビルド方式になる環境では Rust（rustup）が必要
- torchcodec は入れないこと（core 側でシム済み）
- 入れていない場合は自動で標準（spectral）エンジンにフォールバックし、その旨を画面に表示する
