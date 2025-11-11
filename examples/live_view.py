import argparse
import os
import time
from datetime import datetime
from typing import Optional

from edsdk.camera_controller import CameraController


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _try_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _display_with_opencv(cam: CameraController, args) -> None:
    cv2 = _try_import_cv2()
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) が見つかりません。--backend tk で再試行するか、\n"
            "'pip install edsdk-python[display]' または 'pip install -r requirements-examples.txt' を実行してください。"
        )
    try:
        import numpy as np  # type: ignore
    except Exception:
        raise RuntimeError(
            "NumPy が必要です。'pip install edsdk-python[display]' または\n"
            "'pip install numpy' を実行してください。"
        )

    window = args.window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    i = 0
    while True:
        data = cam.grab_live_view_frame()
        # bytes -> OpenCV 画像 (BGR)
        if isinstance(data, str):
            with open(data, "rb") as f:
                raw = f.read()
        else:
            raw = data
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if args.scale and args.scale != 1.0:
            h, w = frame_bgr.shape[:2]
            sw, sh = int(w * args.scale), int(h * args.scale)
            frame_bgr = cv2.resize(frame_bgr, (sw, sh))

        cv2.imshow(window, frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):  # ESC or q
            break
        if key in (ord("s"),):  # snapshot
            if args.snapshot:
                os.makedirs(args.save_dir, exist_ok=True)
                fname = f"{args.prefix}{_now_stamp()}_{i:04d}.jpg"
                out = os.path.join(args.save_dir, fname)
                # すでにBGRなのでそのまま保存
                cv2.imwrite(out, frame_bgr)
                print("Saved:", out)
                i += 1
            else:
                print("Snapshot disabled (use --snapshot to enable)")

    cv2.destroyWindow(window)


def _display_with_tk(cam: CameraController, args) -> None:
    try:
        from PIL import ImageTk  # type: ignore
        import tkinter as tk
    except Exception:  # pragma: no cover - optional path
        raise RuntimeError(
            "Tkinter/Pillow が見つかりません。OpenCV をインストールするか --backend opencv を使用してください。\n"
            "'pip install edsdk-python[display]' または 'pip install Pillow' を実行してください。"
        )

    root = tk.Tk()
    root.title(args.window)
    lbl = tk.Label(root)
    lbl.pack()

    # スナップショット保存用のカウンタ
    i = {"n": 0}

    def on_key(event):
        k = event.keysym.lower()
        if k in ("q", "escape"):
            root.destroy()
        elif k == "s":
            if args.snapshot:
                os.makedirs(args.save_dir, exist_ok=True)
                fname = f"{args.prefix}{_now_stamp()}_{i['n']:04d}.jpg"
                out = os.path.join(args.save_dir, fname)
                cam.grab_live_view_frame(save_path=out)
                print("Saved:", out)
                i["n"] += 1
            else:
                print("Snapshot disabled (use --snapshot to enable)")

    root.bind("<Key>", on_key)

    def update_frame():
        try:
            img = cam.grab_live_view_pil()
            if args.scale and args.scale != 1.0:
                w, h = img.size
                img = img.resize((int(w * args.scale), int(h * args.scale)))
            imgtk = ImageTk.PhotoImage(image=img)
            lbl.imgtk = imgtk  # keep ref
            lbl.configure(image=imgtk)
        except Exception:
            # 取りこぼしはスキップ
            pass
        finally:
            # 約30FPS目安（必要なら --interval を別用途に使わず固定）
            root.after(33, update_frame)

    update_frame()
    root.mainloop()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Canon EDSDK live view")
    p.add_argument("--index", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--save-dir", default=".", help="Directory to save frames")
    p.add_argument("--prefix", default="evf_", help="Filename prefix")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")

    # 保存ループ向け（従来機能）
    p.add_argument(
        "--count", type=int, default=1, help="Number of frames to save (0=until Ctrl+C)"
    )
    p.add_argument(
        "--interval", type=float, default=0.2, help="Interval seconds between frames"
    )

    # 追加: リアルタイム表示
    p.add_argument(
        "--display", action="store_true", help="リアルタイムにウィンドウ表示"
    )
    p.add_argument(
        "--backend",
        choices=["auto", "opencv", "tk"],
        default="auto",
        help="表示バックエンド (auto/opencv/tk)",
    )
    p.add_argument("--window", default="Canon Live View", help="ウィンドウタイトル")
    p.add_argument("--scale", type=float, default=1.0, help="表示スケール (1.0=等倍)")
    p.add_argument(
        "--snapshot",
        action="store_true",
        help="(display時) sキーでスナップショット保存を有効化",
    )

    args = p.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    try:
        with CameraController(
            index=args.index, save_dir=args.save_dir, verbose=args.verbose
        ) as cam:
            cam.start_live_view()

            if args.display:
                # バックエンド選択
                backend = args.backend
                if backend == "auto":
                    # OpenCV優先、なければTk
                    if _try_import_cv2() is not None:
                        _display_with_opencv(cam, args)
                    else:
                        _display_with_tk(cam, args)
                elif backend == "opencv":
                    _display_with_opencv(cam, args)
                else:
                    _display_with_tk(cam, args)
            else:
                # 既存の保存ループ
                i = 0
                while True:
                    path = os.path.join(args.save_dir, f"{args.prefix}{i:06d}.jpg")
                    cam.grab_live_view_frame(save_path=path)
                    print("Saved:", path)
                    i += 1
                    if args.count and i >= args.count:
                        break
                    time.sleep(max(0.0, args.interval))

            cam.stop_live_view()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
