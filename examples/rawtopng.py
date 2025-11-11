import cv2
import rawpy


def RawtoPNG(input_path, output_path):
    print("raw image to PNG image...")
    # raw画像を現像
    rawData = rawpy.imread(input_path)
    # usr_wb = [1700.0, 1024.0, 1346.0, 1024.0]
    # print(rawData.camera_whitebalance)
    rgb = rawData.postprocess(
        use_camera_wb=True,
        no_auto_bright=True,
        bright=7.0,
        gamma=(1.0, 1.0),
    )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, bgr)
    rawData.close()
