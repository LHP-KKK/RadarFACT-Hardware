#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def meters_to_points(m: float) -> float:
    inches = m * 39.37007874015748
    return inches * 72.0

def draw_red_ring_on_canvas(
    img_bgr: np.ndarray,
    board_m: float,
    content_side_m: float,
    ring_gap_m: float = 0.02,
    ring_thickness_m: float = 0.025,
):
    h, w = img_bgr.shape[:2]
    size = min(h, w)
    cx, cy = w // 2, h // 2
    px_per_m = size / board_m

    # --- 核心修改：无视内容区，直接贴边 ---
    # 让圆环的外径距离 1 米板子的边缘留出 1 厘米的空白（防止打印裁切）
    thickness_px = max(1, int(round(ring_thickness_m * px_per_m)))

    # 目标：外径在 0.49m 处（即直径 98cm）
    # r 是圆环中心线的半径，所以 r = 0.49 - (厚度/2)
    r_m = 0.49 - (ring_thickness_m / 2)
    r = int(round(r_m * px_per_m))

    cv2.circle(
        img_bgr,
        (cx, cy),
        r,
        (0, 0, 255),
        thickness=thickness_px,
        lineType=cv2.LINE_AA
    )
    return img_bgr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="out")
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--board_m", type=float, default=1.0)

    # ChArUco config (your 16x16)
    parser.add_argument("--squares", type=int, default=16)
    parser.add_argument("--square_m", type=float, default=0.0625)
    parser.add_argument("--marker_m", type=float, default=0.04375)

    # IMPORTANT: leave a border area for the ring (physical meters)
    parser.add_argument("--border_m", type=float, default=0.12,
                        help="White border margin around ChArUco (meters). 0.10~0.15 recommended for 1m board.")

    # Ring options (physical meters)
    parser.add_argument("--ring", action="store_true")
    parser.add_argument("--ring_gap_m", type=float, default=0.02,
                        help="Gap between ChArUco content and ring inner edge (meters).")
    parser.add_argument("--ring_thickness_m", type=float, default=0.025,
                        help="Ring thickness in meters (e.g., 0.02~0.03).")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Validate geometric consistency of the ChArUco definition (content geometry)
    expected_side_m = args.squares * args.square_m
    # Note: expected_side_m is the physical size of the *ChArUco content*, not the full board now.
    content_side_m = args.board_m - 2 * args.border_m
    if abs(expected_side_m - content_side_m) > 1e-6:
        raise ValueError(
            f"ChArUco content size mismatch.\n"
            f"Your squares*square_m = {expected_side_m:.6f} m\n"
            f"But board_m-2*border_m = {content_side_m:.6f} m\n"
            f"Fix by adjusting one of: squares, square_m, or border_m.\n"
            f"Tip: with 16x16 and square=0.0625m, content_side=1.0m. If you want border, "
            f"you must reduce square_m or squares, OR accept that the ChArUco content becomes smaller than 1m."
        )

    if not (0 < args.marker_m < args.square_m):
        raise ValueError("marker_m must be in (0, square_m).")

    # Pixels for full 1m board
    side_in = args.board_m * 39.37007874015748
    side_px = int(round(side_in * args.dpi))

    # Create blank white canvas for full board
    canvas_bgr = np.full((side_px, side_px, 3), 255, dtype=np.uint8)

    # Generate ChArUco content image at content_side_px
    content_side_px = int(round((content_side_m / args.board_m) * side_px))

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    board = aruco.CharucoBoard((args.squares, args.squares), args.square_m, args.marker_m, dictionary)

    content_gray = board.generateImage((content_side_px, content_side_px))
    content_bgr = cv2.cvtColor(content_gray, cv2.COLOR_GRAY2BGR)

    # Paste content into center
    x0 = (side_px - content_side_px) // 2
    y0 = (side_px - content_side_px) // 2
    canvas_bgr[y0:y0 + content_side_px, x0:x0 + content_side_px] = content_bgr

    # Draw ring on border area (outside content)
    if args.ring:
        canvas_bgr = draw_red_ring_on_canvas(
            canvas_bgr,
            board_m=args.board_m,
            content_side_m=content_side_m,
            ring_gap_m=args.ring_gap_m,
            ring_thickness_m=args.ring_thickness_m,
        )

    # Save
    tag = f"board{args.board_m}m_content{content_side_m:.3f}m_{args.squares}x{args.squares}_{args.dpi}dpi"
    tag = tag.replace(".", "p")
    png_path = outdir / f"charuco_with_border_ring_{tag}.png"
    cv2.imwrite(str(png_path), canvas_bgr)

    pdf_path = outdir / f"charuco_with_border_ring_{tag}.pdf"
    page_side_pt = meters_to_points(args.board_m)
    c = canvas.Canvas(str(pdf_path), pagesize=(page_side_pt, page_side_pt))
    c.drawImage(ImageReader(str(png_path)), 0, 0, width=page_side_pt, height=page_side_pt, mask='auto')
    c.showPage()
    c.save()

    print("Done.")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print("Print at 100% / Actual size (no scaling). No lamination.")


if __name__ == "__main__":
    main()


# python generate_charuco_pdf.py --ring --board_m 1.0 \
#   --squares 9 \
#   --square_m 0.072 \
#   --marker_m 0.05 \
#   --border_m 0.176 \
#   --ring_gap_m 0.02 \
#   --ring_thickness_m 0.02



#   python generate_charuco_pdf.py --ring --board_m 1.0 \
#   --squares 6 --square_m 0.105 --marker_m 0.075 \
#   --border_m 0.185 --ring_gap_m 0.02 --ring_thickness_m 0.02
