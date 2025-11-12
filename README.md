# edsdk-python

Python wrapper for Canon EOS Digital Software Development Kit, aka EDSDK.

Supported Python versions: 3.8 – 3.13 (CPython, Windows 64-bit). Python 3.13 での動作を確認済みです。

Currently, it supports Windows only. But it shouldn't be difficult to adapt it for macOS.

## Obtain the EDSDK from Canon

Before you can use this library you need to obtain the EDSDK library from Canon. You can do so via their developers program:

- [Canon Europe](https://www.canon-europe.com/business/imaging-solutions/sdk/)
- [Canon Americas](https://developercommunity.usa.canon.com)
- [Canon Asia](https://asia.canon/en/campaign/developerresources)
- [Canon Oceania](https://www.canon.com.au/support/support-news/support-news/digital-slr-camera-software-developers-kit)
- [Canon China](https://www.canon.com.cn/supports/sdk/index.html)
- [Canon Korea](https://www.canon-ci.co.kr/support/sdk/sdkMain)
- [Canon Japan](https://cweb.canon.jp/eos/info/api-package/)

Once you were granted access - this may take a few days - download the latest version of their library.

## Copy the EDSDK Headers and Libraries

You should now have access to a zip file containing the file you need to build this library.

Unzip the file and copy the following folders inside the `dependencies` folder of this project:

1. `EDSDK` - Containing headers and 32-bit libraries (we only use the headers!)
2. `EDSDK_64` - Containing 64-bit version of the .lib and .dlls (which we do use!)

Your dependencies folder structure should now look like this:

```text
dependencies/EDSDK/Header/EDSDK.h
dependencies/EDSDK/Header/EDSDKErrors.h
dependencies/EDSDK/Header/EDSDKTypes.h

dependencies/EDSDK_64/Dll/EDSDK.dll
dependencies/EDSDK_64/Dll/EdsImage.dll

dependencies/EDSDK_64/Library/EDSDK.lib
```

Any additional files aren't needed, but won't hurt either in case you copied the entire folders.


## Modify EDSDKTypes.h

This file contains an enum definition, called `Unknown`, which collides with a DEFINE in the `Windows.h` header.

Therefore needs to be renamed.

```c
typedef enum
{
    Unknown   = 0x00000000,
    Jpeg      = 0x3801,
    CR2       = 0xB103,
    MP4       = 0xB982,
    CR3       = 0xB108,
    HEIF_CODE = 0xB10B,
} EdsObjectFormat;
```

You can comment out `Unknown` or rename it to `UNKNOWN` (or whatever you want) or it won't compile on Windows.

## Build the library

Run (this will compile the C++ extension against your current Python, e.g. 3.13):

```cmd
pip install .
```
or
```cmd
pip install .[examples]
or
pip install .[display]
```

To generate a wheel (recommended for distribution / reuse):

```cmd
pip install build
python -m build --wheel
```

You will find the wheel under `dist/` (e.g. `edsdk_python-0.1.1-cp313-cp313-win_amd64.whl`). Install it with:

```cmd
pip install dist\edsdk_python-0.1.1-cp313-cp313-win_amd64.whl
```

## Troubleshooting

If you see errors like:

- C2365: 'Unknown': redefinition; previous definition was 'enumerator'
    Follow [Modify EDSDKTypes.h](#modify-edsdktypesh).

- "OpenCV (cv2) not found" when running examples
    Install extras: `pip install edsdk-python[display]` or `pip install -r requirements-examples.txt`.
