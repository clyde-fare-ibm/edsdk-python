"""Minimal capture example using CameraController.

Release版用の簡潔なサンプル:
  1. セッションを開く
  2. 主要プロパティをまとめて設定
  3. 撮影してファイルパス一覧を表示

カメラの物理モードダイヤルは Manual / Av / Tv など適切なものに合わせてください。
AEモードは多くの機種でSDKから変更不可のため指定していません。
"""

from edsdk.camera_controller import CameraController

# プロパティイベント警告を抑制したい場合は register_property_events=False を指定
with CameraController(
    index=0,
    save_dir="out",
    register_property_events=False,
    file_pattern="{timestamp}_{seq:04d}.{ext}",  # 例: 20251111_153045_0001.jpg
) as cam:
    # 必要ならプロパティをまとめて設定（validateを残すことで不正値を検出）
    cam.set_properties(
        av="8",
        tv="1/15",
        iso=400,
        metering="Evaluative",  # エイリアス対応（内部で EvaluativeMetering にマッピング）
        white_balance="Auto",
        drive_mode="SingleShooting",
        image_quality="LJF",  # 機種により異なります。非対応なら ValueError が発生。
        validate=True,
        tolerate_not_supported=True,  # AEMode 等 read-only の場合にスキップ
    )
    images = cam.capture_pil()
    # show images or process them as needed
    for img in images:
        img.show()

    paths = cam.capture(shots=1, retry=1, retry_delay=0.3)
    for p in paths:
        print("Saved:", p)

    # 完全指定のファイル名で保存（拡張子はカメラ側の種類に合わせて自動付与されます）
    # 例: "my_shot" -> 実際の保存名は "my_shot.JPG" や "my_shot.CR3" など
    fixed = cam.capture(filename="my_shot")
    for p in fixed:
        print("Saved (fixed name):", p)
