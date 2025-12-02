"""Minimal capture example using CameraController.

Release版用の簡潔なサンプル:
  1. セッションを開く
  2. 主要プロパティをまとめて設定
  3. 撮影してファイルパス一覧を表示

カメラの物理モードダイヤルは Manual / Av / Tv など適切なものに合わせてください。
AEモードは多くの機種でSDKから変更不可のため指定していません。
"""

from edsdk.camera_controller import CameraController
import rawtopng
import cv2
import PIL.Image as Image
import numpy as np


def raw_processor(raw_bytes: bytes) -> Image.Image:
    import io
    import rawpy

    print("raw image to PNG image...")
    # raw画像を現像
    with rawpy.imread(io.BytesIO(raw_bytes)) as rawData:
        rgb = rawData.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            bright=5.0,
            gamma=(1.0, 1.0),
        )
    return Image.fromarray(rgb)


# プロパティイベント警告を抑制したい場合は register_property_events=False を指定
with CameraController(
    index=0,
    save_dir=".",
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
        # AF設定（機種・モードにより変更不可のことがあります）
        # 指定例: af_mode="OneShotAF" / "AIServoAF_ServoAF" / "AIFocusAF"
        # マニュアルフォーカスに固定したい場合:
        manual_focus=True,
        # ライブビュー時のAF方式（対応機種のみ）
        # evf_af_mode="Live",         # 例: Live / WholeArea / LiveSpotAF など
        validate=True,
        tolerate_not_supported=True,  # AEMode 等 read-only の場合にスキップ
    )

    # examle1: 撮影して保存、パスを表示
    paths = cam.capture(shots=1, retry=1, retry_delay=0.3)
    for p in paths:
        print("Saved:", p)

    # example2: 撮影してPIL形式で取得、表示
    images = cam.capture_pil()
    # show images or process them as needed
    for img in images:
        img.show()

    # example3: ファイル名を完全指定して保存
    # 完全指定のファイル名で保存（拡張子はカメラ側の種類に合わせて自動付与されます）
    # 例: "my_shot" -> 実際の保存名は "my_shot.JPG" や "my_shot.CR3" など
    fixed = cam.capture(filename="my_shot")
    for p in fixed:
        print("Saved (fixed name):", p)

    # example4: ライブビュー画像取得
    # ライブビュー画像をPIL形式で取得して表示
    live_img = cam.grab_live_view_pil()
    live_img.show()

    # example5: RAW画像をPNGに変換して保存
    cam.set_properties(image_quality="LR")
    raw_paths = cam.capture(shots=1)
    for raw_path in raw_paths:
        png_path = raw_path.rsplit(".", 1)[0] + ".png"
        rawtopng.RawtoPNG(raw_path, png_path)
        print("Converted to PNG:", png_path)

    # example6: 撮影してOpenCV形式で取得、表示
    raw_paths = cam.capture(shots=1)
    for raw_path in raw_paths:
        cv_img = rawtopng.RawtoPNGCv2(raw_path)
        # cv_img のdepthやshapeを確認したい場合:
        print(cv_img.dtype, cv_img.shape)
        cv2.imshow("Captured Image", cv_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    # example7: RAW画像を指定したプロセッサで現像してPIL形式で取得
    pil_imgs = cam.capture_pil(raw_processor=raw_processor)
    for img in pil_imgs:
        bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow("Captured Image (PIL)", bgr_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
