#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hik_capture_charuco.py

Standalone image capture tool for Hikrobot/Hikvision MVS industrial cameras.
Designed for collecting ChArUco calibration images for the uploaded my_calib workflow.

Keys in preview window:
  SPACE / s : save one image
  a         : toggle auto-save mode
  q / ESC   : quit

Example:
  python3 hik_capture_charuco.py --save_dir ./phone_imgs --prefix frame --width 1280 --height 1024 --exposure 12000 --gain 6
  python3 hik_capture_charuco.py --save_dir ./phone_imgs --auto --interval 1.0 --max_images 60
"""

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def add_mvs_python_path():
    """Try common MVS Python wrapper locations on x86_64 and Jetson/aarch64."""
    candidates = [
        os.environ.get("MVCAM_COMMON_RUNENV"),
        "/opt/MVS/Samples/64/Python/MvImport",
        "/opt/MVS/Samples/aarch64/Python/MvImport",
        "/opt/MVS/Samples/armhf/Python/MvImport",
        "/opt/MVS/Samples/Python/MvImport",
        "/opt/MVS/lib/64",
        "/opt/MVS/lib/aarch64",
    ]
    for p in candidates:
        if p and Path(p).exists() and p not in sys.path:
            sys.path.append(p)


add_mvs_python_path()

try:
    from MvCameraControl_class import *  # noqa: F401,F403
except Exception as e:
    print("[ERROR] Cannot import Hikrobot MVS Python wrapper: MvCameraControl_class.py")
    print("        Please install MVS SDK and check wrapper path, for example:")
    print("        export PYTHONPATH=/opt/MVS/Samples/aarch64/Python/MvImport:$PYTHONPATH")
    print("        or")
    print("        export PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:$PYTHONPATH")
    raise e


def check_ret(ret, msg):
    if ret != 0:
        raise RuntimeError(f"{msg} failed, ret=0x{ret:08x}")


def decode_device_name(dev_info):
    try:
        if dev_info.nTLayerType == MV_GIGE_DEVICE:
            name = bytes(dev_info.SpecialInfo.stGigEInfo.chModelName).split(b"\x00", 1)[0].decode(errors="ignore")
            serial = bytes(dev_info.SpecialInfo.stGigEInfo.chSerialNumber).split(b"\x00", 1)[0].decode(errors="ignore")
            ip = dev_info.SpecialInfo.stGigEInfo.nCurrentIp
            ip_str = f"{(ip >> 24) & 255}.{(ip >> 16) & 255}.{(ip >> 8) & 255}.{ip & 255}"
            return f"GigE model={name}, serial={serial}, ip={ip_str}"
        if dev_info.nTLayerType == MV_USB_DEVICE:
            name = bytes(dev_info.SpecialInfo.stUsb3VInfo.chModelName).split(b"\x00", 1)[0].decode(errors="ignore")
            serial = bytes(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber).split(b"\x00", 1)[0].decode(errors="ignore")
            return f"USB3 model={name}, serial={serial}"
    except Exception:
        pass
    return f"Unknown device, layer={dev_info.nTLayerType}"


def get_device(index=0):
    device_list = MV_CC_DEVICE_INFO_LIST()
    tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
    check_ret(ret, "Enum devices")

    if device_list.nDeviceNum == 0:
        raise RuntimeError("No Hikrobot/Hikvision camera found. Check USB3 cable, power, permissions, and MVS Viewer.")

    print(f"[INFO] Found {device_list.nDeviceNum} camera(s):")
    for i in range(device_list.nDeviceNum):
        dev_info = ctypes.cast(device_list.pDeviceInfo[i], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
        print(f"  [{i}] {decode_device_name(dev_info)}")

    if index < 0 or index >= device_list.nDeviceNum:
        raise ValueError(f"Invalid --device_index {index}, available range: 0..{device_list.nDeviceNum - 1}")

    return ctypes.cast(device_list.pDeviceInfo[index], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents


def set_if_supported(cam, setter, name, value, quiet=False):
    try:
        ret = setter(name, value)
        if ret != 0 and not quiet:
            print(f"[WARN] Set {name}={value} failed, ret=0x{ret:08x}")
        return ret
    except Exception as e:
        if not quiet:
            print(f"[WARN] Set {name}={value} exception: {e}")
        return -1


def configure_camera(cam, args, dev_info):
    # GigE optimization, harmlessly skipped for USB cameras.
    if dev_info.nTLayerType == MV_GIGE_DEVICE:
        packet_size = cam.MV_CC_GetOptimalPacketSize()
        if packet_size and packet_size > 0:
            set_if_supported(cam, cam.MV_CC_SetIntValue, "GevSCPSPacketSize", packet_size, quiet=True)

    # For manual calibration image capture: use continuous acquisition, trigger off.
    set_if_supported(cam, cam.MV_CC_SetEnumValue, "AcquisitionMode", MV_ACQ_MODE_CONTINUOUS, quiet=True)
    set_if_supported(cam, cam.MV_CC_SetEnumValue, "TriggerMode", MV_TRIGGER_MODE_OFF, quiet=True)

    # Optional resolution. If unsupported by this camera/ROI state, the SDK returns a warning only.
    if args.width > 0:
        set_if_supported(cam, cam.MV_CC_SetIntValue, "Width", int(args.width))
    if args.height > 0:
        set_if_supported(cam, cam.MV_CC_SetIntValue, "Height", int(args.height))

    # Disable auto exposure/gain for repeatable calibration images unless user asks auto.
    if args.auto_exposure:
        set_if_supported(cam, cam.MV_CC_SetEnumValue, "ExposureAuto", 2, quiet=True)  # Continuous on many MVS cameras
    else:
        set_if_supported(cam, cam.MV_CC_SetEnumValue, "ExposureAuto", 0, quiet=True)  # Off
        if args.exposure > 0:
            set_if_supported(cam, cam.MV_CC_SetFloatValue, "ExposureTime", float(args.exposure))

    if args.auto_gain:
        set_if_supported(cam, cam.MV_CC_SetEnumValue, "GainAuto", 2, quiet=True)
    else:
        set_if_supported(cam, cam.MV_CC_SetEnumValue, "GainAuto", 0, quiet=True)
        if args.gain >= 0:
            set_if_supported(cam, cam.MV_CC_SetFloatValue, "Gain", float(args.gain))

    if args.fps > 0:
        set_if_supported(cam, cam.MV_CC_SetBoolValue, "AcquisitionFrameRateEnable", True, quiet=True)
        set_if_supported(cam, cam.MV_CC_SetFloatValue, "AcquisitionFrameRate", float(args.fps), quiet=True)


def buffer_address(buf):
    """Return integer address from MVS pBufAddr across SDK wrapper variants."""
    if isinstance(buf, int):
        return buf
    value = getattr(buf, "value", None)
    if isinstance(value, int) and value != 0:
        return value
    try:
        return ctypes.addressof(buf.contents)
    except Exception:
        pass
    try:
        value = ctypes.cast(buf, ctypes.c_void_p).value
        if value:
            return int(value)
    except Exception:
        pass
    raise TypeError(f"Cannot get address from pBufAddr of type {type(buf)}")


def assign_src_data(param, pbuf):
    """Assign source buffer to MVS conversion parameter across SDK wrapper variants."""
    try:
        param.pSrcData = pbuf
        return
    except Exception:
        pass
    try:
        param.pSrcData = ctypes.cast(pbuf, ctypes.POINTER(ctypes.c_ubyte))
        return
    except Exception:
        pass
    param.pSrcData = ctypes.c_void_p(buffer_address(pbuf))


def convert_to_bgr(cam, frame):
    """Convert MVS frame buffer to OpenCV BGR uint8 image."""
    info = frame.stFrameInfo
    w, h = int(info.nWidth), int(info.nHeight)
    src_len = int(info.nFrameLen)
    pixel_type = int(info.enPixelType)

    # Fast paths for common already-8-bit formats.
    if pixel_type == PixelType_Gvsp_Mono8:
        arr = np.ctypeslib.as_array((ctypes.c_ubyte * src_len).from_address(buffer_address(frame.pBufAddr)))
        gray = arr.reshape(h, w).copy()
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if pixel_type == PixelType_Gvsp_BGR8_Packed:
        arr = np.ctypeslib.as_array((ctypes.c_ubyte * src_len).from_address(buffer_address(frame.pBufAddr)))
        return arr.reshape(h, w, 3).copy()

    if pixel_type == PixelType_Gvsp_RGB8_Packed:
        arr = np.ctypeslib.as_array((ctypes.c_ubyte * src_len).from_address(buffer_address(frame.pBufAddr)))
        rgb = arr.reshape(h, w, 3).copy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # General path: ask MVS SDK to convert Bayer/YUV/etc. to BGR8.
    dst_size = w * h * 3
    dst_buf = (ctypes.c_ubyte * dst_size)()

    # Different SDK versions expose MV_CC_PIXEL_CONVERT_PARAM or MV_CC_PIXEL_CONVERT_PARAM_EX.
    ParamClass = globals().get("MV_CC_PIXEL_CONVERT_PARAM_EX", None) or globals().get("MV_CC_PIXEL_CONVERT_PARAM")
    if ParamClass is None:
        raise RuntimeError("MVS wrapper lacks MV_CC_PIXEL_CONVERT_PARAM(_EX); cannot convert pixel format.")

    param = ParamClass()
    ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
    param.nWidth = w
    param.nHeight = h
    param.enSrcPixelType = info.enPixelType
    assign_src_data(param, frame.pBufAddr)
    param.nSrcDataLen = src_len
    param.enDstPixelType = PixelType_Gvsp_BGR8_Packed

    # Field names vary slightly across SDK versions.
    if hasattr(param, "pDstBuffer"):
        param.pDstBuffer = ctypes.cast(dst_buf, ctypes.POINTER(ctypes.c_ubyte))
    elif hasattr(param, "pDstBuf"):
        param.pDstBuf = ctypes.cast(dst_buf, ctypes.POINTER(ctypes.c_ubyte))
    else:
        raise RuntimeError("Unknown destination-buffer field in pixel convert parameter.")

    if hasattr(param, "nDstBufferSize"):
        param.nDstBufferSize = dst_size
    elif hasattr(param, "nDstBufSize"):
        param.nDstBufSize = dst_size

    ret = cam.MV_CC_ConvertPixelType(param)
    check_ret(ret, "Convert pixel type")
    img = np.frombuffer(dst_buf, dtype=np.uint8).reshape(h, w, 3).copy()
    return img


def next_index(save_dir: Path, prefix: str, ext: str):
    max_id = 0
    for p in save_dir.glob(f"{prefix}_*.{ext}"):
        try:
            max_id = max(max_id, int(p.stem.split("_")[-1]))
        except Exception:
            pass
    return max_id + 1


def save_image(img, save_dir: Path, prefix: str, idx: int, ext: str, jpg_quality: int):
    out = save_dir / f"{prefix}_{idx:04d}.{ext}"
    if ext.lower() in ("jpg", "jpeg"):
        ok = cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    else:
        ok = cv2.imwrite(str(out), img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {out}")
    print(f"[SAVE] {out}")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default="phone_imgs", help="Directory for calibration images")
    p.add_argument("--device_index", type=int, default=0, help="Camera index shown by MVS enumeration")
    p.add_argument("--prefix", type=str, default="frame")
    p.add_argument("--ext", type=str, default="jpg", choices=["jpg", "png"])
    p.add_argument("--jpg_quality", type=int, default=95)
    p.add_argument("--width", type=int, default=0, help="Optional image width; 0 keeps camera default")
    p.add_argument("--height", type=int, default=0, help="Optional image height; 0 keeps camera default")
    p.add_argument("--exposure", type=float, default=12000.0, help="ExposureTime in us when auto exposure is off")
    p.add_argument("--gain", type=float, default=6.0, help="Gain in dB when auto gain is off; set -1 to skip")
    p.add_argument("--fps", type=float, default=10.0, help="Optional acquisition FPS; 0 skips setting")
    p.add_argument("--auto_exposure", action="store_true", help="Use camera auto exposure")
    p.add_argument("--auto_gain", action="store_true", help="Use camera auto gain")
    p.add_argument("--auto", action="store_true", help="Auto-save images at --interval seconds")
    p.add_argument("--interval", type=float, default=1.0, help="Auto-save interval in seconds")
    p.add_argument("--max_images", type=int, default=0, help="Stop after saving this many images; 0 means unlimited")
    p.add_argument("--no_preview", action="store_true", help="Do not display preview window")
    p.add_argument("--timeout_ms", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    idx = next_index(save_dir, args.prefix, args.ext)

    dev_info = get_device(args.device_index)
    cam = MvCamera()

    try:
        ret = cam.MV_CC_CreateHandle(dev_info)
        check_ret(ret, "Create handle")
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        check_ret(ret, "Open device")

        configure_camera(cam, args, dev_info)

        ret = cam.MV_CC_StartGrabbing()
        check_ret(ret, "Start grabbing")
        print("[INFO] Grabbing started.")
        print("[INFO] SPACE/s: save, a: toggle auto-save, q/ESC: quit")
        print(f"[INFO] Saving to: {save_dir}")

        frame = MV_FRAME_OUT()
        auto_save = bool(args.auto)
        last_save_t = 0.0
        saved = 0

        while True:
            ctypes.memset(ctypes.byref(frame), 0, ctypes.sizeof(frame))
            ret = cam.MV_CC_GetImageBuffer(frame, args.timeout_ms)
            if ret != 0:
                print(f"[WARN] GetImageBuffer timeout/fail ret=0x{ret:08x}")
                continue

            try:
                img = convert_to_bgr(cam, frame)
            finally:
                cam.MV_CC_FreeImageBuffer(frame)

            show = img
            if not args.no_preview:
                # Fit preview to screen-ish size without changing saved image.
                h, w = show.shape[:2]
                scale = min(1.0, 1280.0 / max(w, 1), 800.0 / max(h, 1))
                if scale < 1.0:
                    show = cv2.resize(show, (int(w * scale), int(h * scale)))
                cv2.putText(show, f"saved={saved} next={idx:04d} auto={auto_save}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Hikrobot capture - SPACE/s save, a auto, q quit", show)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            now = time.time()
            do_save = False
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), 32):
                do_save = True
            if key == ord("a"):
                auto_save = not auto_save
                print(f"[INFO] auto_save = {auto_save}")
            if auto_save and (now - last_save_t >= args.interval):
                do_save = True

            if do_save:
                save_image(img, save_dir, args.prefix, idx, args.ext, args.jpg_quality)
                idx += 1
                saved += 1
                last_save_t = now
                if args.max_images > 0 and saved >= args.max_images:
                    print(f"[INFO] Reached max_images={args.max_images}")
                    break

    finally:
        try:
            cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        try:
            cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
