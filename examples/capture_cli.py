import argparse
import os
from typing import Optional

from edsdk.camera_controller import CameraController


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Canon EDSDK capture helper (extended)")
    p.add_argument("--index", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--save-dir", default=".", help="Directory to save images")
    p.add_argument("--av", help="Aperture value (e.g., 5.6 or f/5.6)")
    p.add_argument("--tv", help="Shutter speed (e.g., 1/125 or 0.5)")
    p.add_argument("--iso", help="ISO value (e.g., 100, 400, auto)")
    p.add_argument("--ae-mode", help="AE mode enum name (e.g., Manual, Av, Tv)")
    p.add_argument("--metering", help="Metering mode enum name")
    p.add_argument("--white-balance", help="White balance enum name")
    p.add_argument("--image-quality", help="Image quality enum name")
    p.add_argument("--drive-mode", help="Drive mode enum name")
    p.add_argument("--shots", type=int, default=1, help="Number of shots to take")
    p.add_argument(
        "--interval", type=float, default=0.0, help="Interval between shots seconds"
    )
    p.add_argument(
        "--retry", type=int, default=0, help="Retry count per shot on timeout"
    )
    p.add_argument(
        "--retry-delay", type=float, default=0.3, help="Delay between retries seconds"
    )
    p.add_argument(
        "--timeout", type=float, default=5.0, help="Timeout per shot seconds"
    )
    p.add_argument(
        "--list", action="store_true", help="List camera supported values and exit"
    )
    p.add_argument(
        "--live-view-frame", help="Grab one live view frame (path to save JPEG)"
    )
    p.add_argument(
        "--save-profile", help="Save current properties to JSON file and exit"
    )
    p.add_argument("--load-profile", help="Load properties from JSON file and apply")
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Do not validate property values against camera supported list",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    try:
        with CameraController(
            index=args.index, save_dir=args.save_dir, verbose=args.verbose
        ) as cam:
            if args.list:
                supported = cam.list_supported()
                for k, vals in supported.items():
                    print(f"{k}: {', '.join(vals)}")
                return 0

            if args.load_profile:
                cam.load_profile(
                    args.load_profile, apply=True, validate=not args.no_validate
                )

            cam.set_properties(
                av=args.av,
                tv=args.tv,
                iso=args.iso,
                ae_mode=args.ae_mode,
                metering=args.metering,
                white_balance=args.white_balance,
                image_quality=args.image_quality,
                drive_mode=args.drive_mode,
            )
            props = cam.get_properties()
            print("Current properties:", props)

            if args.save_profile:
                cam.save_profile(args.save_profile)
                return 0

            if args.live_view_frame:
                path = cam.grab_live_view_frame(save_path=args.live_view_frame)
                print("Live view saved:", path)
                return 0

            paths = cam.capture(
                shots=args.shots,
                timeout=args.timeout,
                interval=args.interval,
                retry=args.retry,
                retry_delay=args.retry_delay,
            )
            for pth in paths:
                print("Saved:", pth)
    except Exception as e:
        try:
            from edsdk.camera_controller import classify_error

            info = classify_error(e)
            if "code" in info:
                print(f"Error: {info.get('message')} (code={info.get('code')})")
            else:
                print(f"Error: {info.get('message')}")
        except Exception:
            print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
