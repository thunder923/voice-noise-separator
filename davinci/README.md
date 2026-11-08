# DaVinci Resolve 連携（Phase 1）

DaVinci Resolve Studio のタイムラインで選択したクリップを、ローカルの
声/ノイズ分離ツールで分離し、VOICE / NOISE ステムを新しい音声トラックに
同じ位置で並べます。Phase 2（VST3プラグイン）より前の連携方式です。

## 必要なもの
- DaVinci Resolve **Studio**（スクリプトAPIはStudio版のみ）
  - 選択クリップの取得に `Timeline.GetSelectedClips()` を使うため
    **Resolve 21.0.3 以降**を推奨。それ未満では再生ヘッド下のクリップで代替
- ffmpeg（PATH が通っていること）
- 声/ノイズ分離アプリがローカルで起動していること
  （音声ポータル、または単体起動。既定 http://127.0.0.1:8200）

## インストール
`VoiceNoiseSeparate.py` を Resolve のスクリプトフォルダへコピー:
- macOS: `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Support/Fusion/Scripts/Edit/`
- Windows: `%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Edit\`

Resolve のメニュー「ワークスペース > スクリプト > Edit > VoiceNoiseSeparate」
から実行します。

## 使い方
1. 分離したいクリップをタイムラインで選択（動画クリップを選ぶとリンクした音声が対象になります）
2. スクリプトを実行
3. `VoiceNoise VOICE` / `VoiceNoise NOISE` トラックにステムが配置されます

## 設定
同じフォルダに `voice_noise_config.json` を置くと上書きできます:
`server` / `engine`（spectral・deepfilternet・mossformer2・reuse・dbs・bandit）/
`strength` / `natural` / `fix_outdoor` / `fix_ambience` / `loudness_preset` /
`ffmpeg` / `dry_run`（true で配置せず計画だけ表示）

## v1 の制限
- 速度変更（リタイム）したクリップ、ネストしたタイムラインは未対応
  （素材のフレームレートで in/out を取る前提）
- 処理はクリップ単位。長いクリップはそのまま時間がかかります
