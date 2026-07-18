#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from pathlib import Path
import numpy as np


def parse_scalar(text, key, cast=float, default=None):
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*([^\n#]+)", text, re.MULTILINE)
    if not m:
        if default is not None:
            return default
        raise RuntimeError(f"Missing key: {key}")
    v = m.group(1).strip()
    if cast is str:
        return v
    return cast(v)


def parse_vector(text, key, expected_len):
    m = re.search(rf"{re.escape(key)}\s*:\s*\[([^\]]+)\]", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"Missing vector/matrix key: {key}")
    nums = re.findall(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
        m.group(1),
    )
    vals = [float(x) for x in nums]
    if len(vals) != expected_len:
        raise RuntimeError(f"{key} expects {expected_len} numbers, got {len(vals)}")
    return vals


def inv_rt(R, t):
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    Ri = R.T
    ti = -Ri @ t
    return Ri, ti


def mat_to_list(R):
    return np.asarray(R, dtype=float).reshape(3, 3).tolist()


def vec_to_list(t):
    return np.asarray(t, dtype=float).reshape(3).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="ÀýÈç ~/iq1m_demo/indoor_forward_03_trim.fwd")
    ap.add_argument("--raw-calib", required=True, help="FAST-LIVO2 calib_result.txt")
    ap.add_argument(
        "--direction",
        default="lidar_to_camera",
        choices=["lidar_to_camera", "camera_to_lidar"],
        help=(
            "Ä¬ÈÏ¼ÙÉè calib_result.txt µÄ Rcl/Pcl ±íÊ¾ p_camera = Rcl * p_lidar + Pcl¡£"
            "Èç¹ûºóÐøÍ¶Ó°Ã÷ÏÔ²»¶Ô£¬ÔÙÓÃ camera_to_lidar ÖØÅÜ¡£"
        ),
    )
    ap.add_argument("--calibration-date", default="")
    ap.add_argument("--method", default="FAST-LIVO2 / camera-lidar calibration")
    args = ap.parse_args()

    scene = Path(args.scene).expanduser().resolve()
    raw_calib = Path(args.raw_calib).expanduser().resolve()

    if not scene.exists():
        raise FileNotFoundError(f"scene not found: {scene}")
    if not raw_calib.exists():
        raise FileNotFoundError(f"calib_result.txt not found: {raw_calib}")

    text = raw_calib.read_text(errors="ignore")

    cam_model = parse_scalar(text, "cam_model", str)
    width = parse_scalar(text, "cam_width", int)
    height = parse_scalar(text, "cam_height", int)
    scale = parse_scalar(text, "scale", float, default=1.0)

    fx = parse_scalar(text, "cam_fx", float)
    fy = parse_scalar(text, "cam_fy", float)
    cx = parse_scalar(text, "cam_cx", float)
    cy = parse_scalar(text, "cam_cy", float)

    d = [
        parse_scalar(text, "cam_d0", float),
        parse_scalar(text, "cam_d1", float),
        parse_scalar(text, "cam_d2", float),
        parse_scalar(text, "cam_d3", float),
    ]

    Rcl = np.array(parse_vector(text, "Rcl", 9), dtype=float).reshape(3, 3)
    Pcl = np.array(parse_vector(text, "Pcl", 3), dtype=float).reshape(3)

    # Õý½»ÐÔ¼ì²é
    det_R = float(np.linalg.det(Rcl))
    ortho_err = float(np.linalg.norm(Rcl.T @ Rcl - np.eye(3)))

    if args.direction == "lidar_to_camera":
        R_lidar_to_camera = Rcl
        t_lidar_to_camera = Pcl
        assumed_note = "Assumed from calib_result: p_camera = Rcl @ p_lidar + Pcl"
    else:
        # ¼ÙÉèÔ­ÎÄ¼þÊÇ p_lidar = Rcl @ p_camera + Pcl£¬Ôò·´½âµÃµ½ LiDAR -> Camera
        R_lidar_to_camera, t_lidar_to_camera = inv_rt(Rcl, Pcl)
        assumed_note = "Assumed from calib_result: p_lidar = Rcl @ p_camera + Pcl; inverted to lidar_to_camera"

    R_camera_to_lidar, t_camera_to_lidar = inv_rt(R_lidar_to_camera, t_lidar_to_camera)

    K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]

    scene_calib = scene / "_calib"
    cam_dir = scene / "_camera"
    cam_pub_dir = scene / "camera"
    lidar_dir = scene / "lidar"
    hidden_lidar_dir = scene / "_lidar"

    for p in [scene_calib, cam_dir, cam_pub_dir, lidar_dir, hidden_lidar_dir]:
        p.mkdir(parents=True, exist_ok=True)

    shutil.copy2(raw_calib, scene_calib / "calib_result.txt")

    raw_cam_json = {
        "image_width": width,
        "image_height": height,
        "camera_model": cam_model.lower(),
        "scale": scale,
        "K": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "K_matrix": K,
        "distortion": {
            "model": "plumb_bob",
            "coefficients": d,
            "order": ["k1", "k2", "p1", "p2"],
        },
        "undistortion_applied": False,
        "original_calib_path": str(raw_calib),
        "notes": "Raw camera intrinsics parsed from FAST-LIVO2 calib_result.txt.",
    }

    # ³õÊ¼ calib.json ÏÈÐ´ raw£»Stage B È¥»û±äºó»á¸üÐÂÎª undistorted K + zero distortion
    for dst in [
        cam_dir / "calib_raw.json",
        cam_dir / "calib.json",
        cam_pub_dir / "calib_raw.json",
        cam_pub_dir / "calib.json",
    ]:
        dst.write_text(json.dumps(raw_cam_json, indent=2, ensure_ascii=False))

    extr_l2c = {
        "source_sensor": "livox_mid360",
        "target_sensor": "camera",
        "transform_direction": "lidar_to_camera",
        "R": mat_to_list(R_lidar_to_camera),
        "t": vec_to_list(t_lidar_to_camera),
        "T_4x4": np.vstack([
            np.hstack([R_lidar_to_camera, t_lidar_to_camera.reshape(3, 1)]),
            np.array([[0.0, 0.0, 0.0, 1.0]])
        ]).tolist(),
        "t_unit": "meters",
        "calibration_method": args.method,
        "calibration_date": args.calibration_date,
        "original_calib_path": str(raw_calib),
        "original_keys": {"R": "Rcl", "t": "Pcl"},
        "assumption": assumed_note,
        "Rcl_determinant": det_R,
        "Rcl_orthogonality_error_fro": ortho_err,
        "notes": (
            "If LiDAR projection QA is mirrored, behind camera, or globally inconsistent, "
            "rerun this script with --direction camera_to_lidar and redo projection QA."
        ),
    }

    extr_c2l = {
        "source_sensor": "camera",
        "target_sensor": "livox_mid360",
        "transform_direction": "camera_to_lidar",
        "R": mat_to_list(R_camera_to_lidar),
        "t": vec_to_list(t_camera_to_lidar),
        "T_4x4": np.vstack([
            np.hstack([R_camera_to_lidar, t_camera_to_lidar.reshape(3, 1)]),
            np.array([[0.0, 0.0, 0.0, 1.0]])
        ]).tolist(),
        "t_unit": "meters",
        "calibration_method": args.method,
        "calibration_date": args.calibration_date,
        "derived_from": "inverse of lidar_to_camera",
    }

    for dst in [
        cam_dir / "extrinsics_lidar.json",
        cam_pub_dir / "extrinsics_lidar.json",
        lidar_dir / "extrinsics_camera.json",
        hidden_lidar_dir / "extrinsics_camera.json",
    ]:
        dst.write_text(json.dumps(extr_l2c, indent=2, ensure_ascii=False))

    (scene_calib / "extrinsics_camera_to_lidar.json").write_text(
        json.dumps(extr_c2l, indent=2, ensure_ascii=False)
    )

    print("[OK] installed calibration into scene:")
    print(" scene:", scene)
    print(" raw calib:", raw_calib)
    print(" wrote:", cam_dir / "calib_raw.json")
    print(" wrote:", cam_dir / "calib.json")
    print(" wrote:", cam_dir / "extrinsics_lidar.json")
    print(" wrote:", lidar_dir / "extrinsics_camera.json")
    print()
    print("Rcl determinant:", det_R)
    print("Rcl orthogonality error:", ortho_err)
    print("direction:", args.direction)
    print("assumption:", assumed_note)


if __name__ == "__main__":
    main()
