import argparse
import os
import time
from typing import Optional

from edsdk.camera_controller import CameraController


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Canon EDSDK live view grabber")
    p.add_argument("--index", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--save-dir", default=".", help="Directory to save frames")
    p.add_argument(
        "--count", type=int, default=1, help="Number of frames to save (0=until Ctrl+C)"
    )
    p.add_argument(
        "--interval", type=float, default=0.2, help="Interval seconds between frames"
    )
    p.add_argument("--prefix", default="evf_", help="Filename prefix")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    i = 0
    try:
        with CameraController(
            index=args.index, save_dir=args.save_dir, verbose=args.verbose
        ) as cam:
            cam.start_live_view()
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
