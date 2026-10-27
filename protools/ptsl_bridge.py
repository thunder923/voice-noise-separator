"""Pro Tools bridge via PTSL (Pro Tools Scripting Library).

What it does:
  1. Separates a clip (or takes already-separated *_VOICE.wav / *_NOISE.wav)
  2. Connects to the *currently open* Pro Tools session via py-ptsl (gRPC)
  3. Imports both stems and creates two tracks:  <name>_VOICE / <name>_NOISE
     so the editor can re-balance dialogue vs. ambience with two faders.

Requirements:
  - Pro Tools 2022.12 or later (Scripting SDK / PTSL is built in, free)
  - pip install py-ptsl
  - Pro Tools running with the target session open

If py-ptsl / Pro Tools is not available, it still does the separation and
prints the two file paths for manual drag-import — it never fails silently.

Usage:
  python ptsl_bridge.py /path/to/clip.wav [--strength 0.8] [--engine spectral]
  python ptsl_bridge.py --import-only clip_VOICE.wav clip_NOISE.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.separator import separate_file


def import_to_protools(voice_path: Path, noise_path: Path, base_name: str) -> bool:
    try:
        from ptsl import open_engine
    except Exception:
        print("[PTSL] py-ptsl が入っていません: pip install py-ptsl")
        return False
    try:
        with open_engine(company_name="Kanta", application_name="VoiceNoiseSeparator") as engine:
            print(f"[PTSL] セッション接続: {engine.session_name()}")
            # Import audio files into the session's clip list
            engine.import_audio([str(voice_path), str(noise_path)])
            # Create the two destination tracks (mono/stereo follows the session)
            engine.create_new_tracks(
                number_of_tracks=2,
                track_name_list=[f"{base_name}_VOICE", f"{base_name}_NOISE"],
            )
            print("[PTSL] インポート完了。クリップリストから各トラックへ配置してください"
                  "（配置の自動化は PTSL のバージョンにより提供コマンドが異なるため、"
                  "次回Pro Tools実機で確認してから有効化します）")
        return True
    except Exception as e:
        print(f"[PTSL] Pro Tools に接続できませんでした（起動してセッションを開いていますか？）: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="clip.wav 1つ（分離から） / --import-only 時は VOICE と NOISE の2つ")
    ap.add_argument("--import-only", action="store_true")
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--engine", default="spectral")
    args = ap.parse_args()

    if args.import_only:
        voice, noise = Path(args.inputs[0]), Path(args.inputs[1])
        base = voice.stem.replace("_VOICE", "")
    else:
        src = Path(args.inputs[0])
        info = separate_file(src, src.parent / "separated",
                             engine=args.engine, strength=args.strength)
        voice, noise = Path(info["voice_path"]), Path(info["noise_path"])
        base = src.stem
        print(f"分離完了: {voice} / {noise} (engine={info['engine_used']})")

    ok = import_to_protools(voice, noise, base)
    if not ok:
        print("手動インポート用ファイル:")
        print(f"  VOICE: {voice}")
        print(f"  NOISE: {noise}")


if __name__ == "__main__":
    main()
