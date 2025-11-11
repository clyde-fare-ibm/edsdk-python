import os
import time
import uuid

import edsdk
from edsdk import (
    CameraCommand,
    ObjectEvent,
    PropID,
    FileCreateDisposition,
    Access,
    SaveTo,
    EdsObject,
    PropertyEvent,
)
from edsdk.constants.properties import Av, Tv, ISOSpeedCamera

if os.name == "nt":
    # If you're using the EDSDK on Windows,
    # you have to have a Windows message loop in your main thread,
    # otherwise callbacks won't happen.
    # (This is because the EDSDK uses the obsolete COM STA threading model
    # instead of real threads.)
    import pythoncom


def save_image(object_handle: EdsObject, save_to: str) -> int:
    info = edsdk.GetDirectoryItemInfo(object_handle)
    filename = info.get("szFileName") or f"{uuid.uuid4()}.bin"
    dst = os.path.join(save_to, filename)
    out_stream = edsdk.CreateFileStream(
        dst,
        FileCreateDisposition.CreateAlways,
        Access.ReadWrite,
    )
    edsdk.Download(object_handle, info["size"], out_stream)
    edsdk.DownloadComplete(object_handle)
    print(f"Saved: {dst}")
    return 0


def callback_property(event: PropertyEvent, property_id: PropID, parameter: int) -> int:
    print("event: ", event)
    print("Property changed:", property_id)
    print("Parameter:", parameter)
    return 0


def callback_object(event: ObjectEvent, object_handle: EdsObject) -> int:
    print("event: ", event, "object_handle:", object_handle)
    if event == ObjectEvent.DirItemRequestTransfer:
        save_image(object_handle, ".")
    return 0


if __name__ == "__main__":
    edsdk.InitializeSDK()
    cam_list = edsdk.GetCameraList()
    nr_cameras = edsdk.GetChildCount(cam_list)

    if nr_cameras == 0:
        print("No cameras connected")
        exit(1)

    cam = edsdk.GetChildAtIndex(cam_list, 0)
    edsdk.OpenSession(cam)
    edsdk.SetObjectEventHandler(cam, ObjectEvent.All, callback_object)
    edsdk.SetPropertyData(cam, PropID.SaveTo, 0, SaveTo.Host)

    edsdk.SetPropertyData(cam, PropID.Av, 0, 0x38)
    edsdk.SetPropertyData(cam, PropID.Tv, 0, 0x48)
    edsdk.SetPropertyData(
        cam, PropID.ISOSpeed, 0, ISOSpeedCamera.ISO400
    )  # e.g., set to "100"
    print(edsdk.GetPropertyData(cam, PropID.SaveTo, 0))
    print(Av[edsdk.GetPropertyData(cam, PropID.Av, 0)])
    print(Tv[edsdk.GetPropertyData(cam, PropID.Tv, 0)])
    print()
    # Sets HD Capacity to an arbitrary big value
    edsdk.SetCapacity(
        cam, {"reset": True, "bytesPerSector": 512, "numberOfFreeClusters": 2147483647}
    )
    print(edsdk.GetDeviceInfo(cam))

    edsdk.SendCommand(cam, CameraCommand.TakePicture, 0)

    time.sleep(4)
    if os.name == "nt":
        pythoncom.PumpWaitingMessages()
    edsdk.TerminateSDK()
