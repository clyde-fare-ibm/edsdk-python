# Capture CLI usage

このフォルダの `capture_cli.py` は、より扱いやすい撮影ラッパー `edsdk.camera_controller.CameraController` を使ったサンプルです。

使い方（Windows, cmd）:

```cmd
python examples\capture_cli.py --av 5.6 --tv 1/125 --iso 400 --save-dir out --shots 1
```

- `--av`: f値（例: `5.6`, `f/8`）
- `--tv`: シャッター速度（例: `1/125`, `0.5`）
- `--iso`: ISO（数値または `auto`）
- `--save-dir`: 保存先ディレクトリ（標準はカレントディレクトリ）
- `--shots`: 連続撮影枚数（標準 1）

主な追加オプション:

- `--ae-mode`: AEモード（例: `Manual`, `Av`, `Tv`）
- `--metering`: 測光モード（例: `Evaluative`, `CenterWeightedAverage`, `Spot`）
- `--white-balance`: ホワイトバランス（例: `Auto`, `Daylight`, `Cloudy`）
- `--image-quality`: 画質（例: `LJ`, `M1`, `RAW` などモデル依存）
- `--drive-mode`: ドライブモード（例: `SingleShooting`, `ContinuousShooting`）
- `--interval`: 連写間隔（秒）
- `--retry` / `--retry-delay`: タイムアウト時のリトライ回数/間隔
- `--timeout`: 1枚あたりの転送待ちタイムアウト（秒）
- `--list`: カメラが受け付ける候補値を一覧表示して終了
- `--live-view-frame`: ライブビュー1フレームをJPEGで保存して終了
- `--save-profile`: 現在のプロパティをJSONに保存して終了
- `--load-profile`: JSONからプロパティを読み込み・適用
- `--no-validate`: 候補チェックをスキップして強制適用（通常は不要）

注意:

- Windows では EDSDK のコールバック処理のため、内部でメッセージポンプを回しています。撮影後の転送完了まで `--timeout` 秒（標準 5s）待機します。
- カメラのモードダイヤル・機種により設定可能な値は異なります。無効な値を指定するとエラーになります。

`CameraController` は `with` 構文でセッションの開始/終了を自動管理し、`set_properties()` と `capture()` を提供します。

```python
from edsdk.camera_controller import CameraController

with CameraController(index=0, save_dir="out") as cam:
    cam.set_properties(
        av="f/8", tv="1/125", iso=400,
        ae_mode="Manual", metering="Evaluative", white_balance="Auto",
        image_quality="LJ", drive_mode="SingleShooting",
    )
    # 候補の取得
    print(cam.list_supported())

    # 撮影（間隔・リトライ指定）
    paths = cam.capture(shots=3, interval=0.5, retry=1, retry_delay=0.3)
    print(paths)
```

ライブビューの単発取得:

```python
with CameraController(index=0, save_dir="out") as cam:
    cam.start_live_view()
    cam.grab_live_view_frame(save_path="out/evf.jpg")
    cam.stop_live_view()
```

CLI 拡張の例:

```cmd
# 候補表示
python examples\capture_cli.py --list

# 追加プロパティ設定や間隔・リトライ
python examples\capture_cli.py --av 5.6 --tv 1/125 --iso 400 --ae-mode Manual --metering Evaluative --shots 3 --interval 0.5 --retry 1

# ライブビュー1フレーム保存
python examples\capture_cli.py --live-view-frame out\evf.jpg

# 連続ライブビュー
python examples\live_view.py --save-dir out\evf --count 10 --interval 0.2
```

プロファイル（JSON）を使ったワークフロー:

```cmd
# 今の設定を保存
python examples\capture_cli.py --save-profile config\studioA.json

# あとで読み込んで適用（撮影せず適用のみしたい場合は --shots 0 相当の今後の拡張を検討）
python examples\capture_cli.py --load-profile config\studioA.json --shots 1 --save-dir out

# 候補チェックを無効化して適用（通常は推奨しません）
python examples\capture_cli.py --load-profile config\studioA.json --no-validate
```

補足:

- 例外発生時は EDSDK のエラーコードを含むメッセージで表示されます（`classify_error` を内部利用）。
- Python から直接使う場合は、`CameraController.enable_async()` と `pump_events()` を使って非同期でイベント（撮影完了など）を受け取ることもできます。
