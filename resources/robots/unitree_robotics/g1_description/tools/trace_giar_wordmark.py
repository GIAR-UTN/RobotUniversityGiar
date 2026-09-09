"""Trace the real GIAR wordmark out of a brand image into a 3D mesh.

Unlike generate_logo_mesh.py's title text (a stock font run through Blender),
this reproduces the actual custom letterforms of the GIAR logo (flat-topped
A, split "i" dot, angular G) by thresholding the brand image, extracting
letter silhouettes (with their holes -- the R's bowl, etc.) via OpenCV
contours, and extruding them with trimesh. No Blender/GUI needed.

Usage:
    .venv/bin/python resources/robots/unitree_robotics/g1_description/tools/trace_giar_wordmark.py \
        --source web/assets/giar-logo.png --crop 955,85,1310,210 \
        --out /tmp/giar_wordmark_title.STL

`--crop LEFT,TOP,RIGHT,BOTTOM` (pixels) should tightly bound just the "GIAR"
wordmark, no subtitle/tagline text and no stray background noise -- inspect
`--debug-mask` output first to check threshold quality before trusting the
traced mesh.

Output convention: lies flat in the local XY plane (Y = width, X = height,
image Y-down flipped to make X point up), extruded a small amount along Z
(the thin "depth" axis) -- i.e. NOT yet in generate_logo_mesh.py's shared
+X-depth/+Y-width/+Z-height convention. Integrating this into that pipeline
(as a drop-in replacement for the Blender-text title) still needs a rotation
+ rescale step to match the subtitle's frame -- not done by this script.
"""
import argparse

import cv2
import numpy as np
import trimesh
from shapely.geometry import Polygon


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--crop", required=True, help="LEFT,TOP,RIGHT,BOTTOM in pixels")
    p.add_argument("--threshold", type=int, default=150)
    p.add_argument("--simplify", type=float, default=1.0,
                   help="cv2.approxPolyDP epsilon in pixels; 0 to disable")
    p.add_argument("--depth", type=float, default=0.005, help="extrusion depth, meters")
    p.add_argument("--target-height", type=float, default=0.03,
                   help="final wordmark height in meters")
    p.add_argument("--out", required=True)
    p.add_argument("--debug-mask", default=None, help="optional path to dump the threshold mask")
    return p.parse_args()


def build_polygons(mask, simplify_eps):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]  # (N, [next, prev, first_child, parent])

    def simplify(c):
        if simplify_eps <= 0:
            return c
        return cv2.approxPolyDP(c, simplify_eps, closed=True)

    polygons = []
    for i, h in enumerate(hierarchy):
        parent = h[3]
        if parent != -1:
            continue  # holes are handled as part of their parent, below
        if cv2.contourArea(contours[i]) < 20:
            continue  # discard degenerate/noise contours
        shell = simplify(contours[i]).reshape(-1, 2)
        if len(shell) < 3:
            continue
        holes = []
        child = h[2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= 20:
                hole_pts = simplify(contours[child]).reshape(-1, 2)
                if len(hole_pts) >= 3:
                    holes.append(hole_pts)
            child = hierarchy[child][0]
        polygons.append(Polygon(shell, holes))
    return polygons


def main():
    args = parse_args()
    left, top, right, bottom = (int(v) for v in args.crop.split(","))

    img = cv2.imread(args.source)
    if img is None:
        raise SystemExit(f"could not read {args.source}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crop = gray[top:bottom, left:right]

    _, mask = cv2.threshold(crop, args.threshold, 255, cv2.THRESH_BINARY)
    if args.debug_mask:
        cv2.imwrite(args.debug_mask, mask)

    polygons = build_polygons(mask, args.simplify)
    print(f"[trace_giar_wordmark] traced {len(polygons)} letter shapes")

    meshes = [trimesh.creation.extrude_polygon(poly, height=args.depth) for poly in polygons]
    combined = trimesh.util.concatenate(meshes)

    # image space: x=right, y=DOWN. Flip y so it points up, then scale X/Y
    # (still in pixels) so the wordmark's pixel height maps to
    # --target-height meters -- Z is already real meters (extrude_polygon's
    # `height=args.depth`), so it must NOT be scaled by the same factor.
    # Mirroring a single axis is an improper transform -- it flips every
    # triangle's winding, turning extrude_polygon's correct outward-facing
    # normals inside-out (negative signed volume). Without invert(), the
    # side walls render as back-face-culled at any raking angle: you see
    # straight through the letter to whatever's behind it ("hollow" look).
    combined.vertices[:, 1] *= -1
    combined.invert()
    bounds = combined.bounds  # (2,3): min/max per axis
    pixel_height = bounds[1][1] - bounds[0][1]
    scale = args.target_height / pixel_height
    combined.vertices[:, :2] *= scale
    combined.vertices -= combined.bounds.mean(axis=0)

    combined.export(args.out)
    size = combined.bounds[1] - combined.bounds[0]
    print(f"[trace_giar_wordmark] final size (x=width, y=height, z=depth): {size}")
    print(f"[trace_giar_wordmark] exported to {args.out}")


if __name__ == "__main__":
    main()
