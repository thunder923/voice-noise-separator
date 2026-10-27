# プラグイン版（VST3 / AU / AAX）

1つのプラグインの中に3モード:
- **Voice Only** … 声だけにするツール
- **Noise Only** … ノイズだけにするツール（原音から声ゲインを引いた残差）
- **Blend** … 声ゲイン／ノイズゲインを別々に動かしてバランス調整

中身は Web/Mac版コアと同じ原理（ブロック最小統計ノイズ推定＋Wiener系ソフトマスク）の
リアルタイム版。レイテンシは FFT 2048 / hop 512 で約43ms（オフライン処理のAudioSuite用途が主想定、
ライブモニター用途ではない）。

## ビルド
```
git clone https://github.com/juce-framework/JUCE plugin/JUCE   # このフォルダの隣に JUCE を置く
cd plugin && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```
VST3 / AU はそのままビルドできる（Logic等でも使える）。

## Pro Tools（AAX）の現実 — 先に知っておくべきこと
- AAX は「コンパイルすれば終わり」ではない。**Avid の開発者承認＋AAX SDK** が必要。
- **製品版 Pro Tools で読み込ませるには PACE Eden による署名が必須**で、署名鍵は
  **物理 iLok ドングル**の中にある（audiosdk@avid.com へ申請）。
- 署名なしAAXは無料の **Pro Tools Developer** ビルドでのみ読み込める（テスト用）。
- つまり順番は: ①VST3/AUで完成 → ②Developer版Pro ToolsでAAXテスト →
  ③Avid承認＋iLok署名 → ④製品版で配布可能、になる。
- それまでの間、Pro Tools との実用連携は `../protools/ptsl_bridge.py`（PTSL経由の
  自動インポート）が担う。こちらは承認不要・無料SDKだけで今日から使える。

## 状態
このプラグインのソースは土台（スケルトン＋処理部）まで用意済み。
**このVM（Linux・JUCE未導入）ではコンパイル検証は未実施** — Mac上で JUCE を入れて
初回ビルド確認をするのが次のステップ。
