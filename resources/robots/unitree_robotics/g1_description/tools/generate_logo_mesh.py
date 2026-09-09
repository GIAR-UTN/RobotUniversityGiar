"""Generate the G1 chest/back emblem mesh (title + subtitle) as N colored STLs.

Runs headless inside Blender -- no GUI, no MCP addon needed:

    /Applications/Blender.app/Contents/MacOS/Blender --background --python \
        resources/robots/unitree_robotics/g1_description/tools/generate_logo_mesh.py -- \
        --title GIAR --subtitle "ruGIAR stack" \
        --range 0:2:accent --range 7:12:black \
        --out main:resources/robots/unitree_robotics/g1_description/meshes/giar_logo_link.STL \
        --out accent:resources/robots/unitree_robotics/g1_description/meshes/giar_logo_link_accent.STL \
        --out black:resources/robots/unitree_robotics/g1_description/meshes/giar_logo_link_black.STL

`--range START:END:NAME` assigns subtitle characters [START, END) (0-indexed,
end-exclusive) to a named color group -- e.g. "ruGIAR stack" with
`--range 0:2:accent --range 7:12:black` splits "ru" into "accent" and
"stack" into "black", leaving "GIAR stack"'s "GIAR" (and the whole title) in
the default "main" group. Pass one `--out NAME:PATH` per group that should
be written (a group with no matching `--out` is still generated but not
exported -- harmless, just skip listing it). apply_logo_to_models.py then
rgba's each exported piece independently.

Output convention (matches what apply_logo_to_models.py expects): every
exported piece shares the same local coordinate frame -- centered on the
*combined* geometry's own bounding-box center at 0,0,0 -- and is oriented
with local +X as the thin "outward normal" axis, +Y as text width, +Z as
text height. apply_logo_to_models.py places them via each model's own geom
<pos>/<quat>; this script only controls the emblem's shape, never where it
sits on the robot.
"""
import argparse
import sys

import bpy


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--title", default="GIAR")
    p.add_argument("--title-mesh", default=None,
                   help="STL to use for the title instead of rendering --title as text "
                        "(see trace_giar_wordmark.py) -- must already be X=width/Y=height-up/"
                        "Z=depth oriented, e.g. that script's own output convention")
    p.add_argument("--subtitle", default="ruGIAR stack")
    p.add_argument("--range", action="append", default=[], dest="ranges",
                   help="START:END:NAME -- subtitle char range [START,END) to color group NAME")
    p.add_argument("--out", action="append", default=[], dest="outs", required=True,
                   help="NAME:PATH -- export the named color group's mesh to PATH")
    p.add_argument("--title-size", type=float, default=1.4)
    p.add_argument("--subtitle-size", type=float, default=0.6)
    p.add_argument("--title-extrude", type=float, default=0.06)
    p.add_argument("--title-bevel", type=float, default=0.015)
    p.add_argument("--subtitle-extrude", type=float, default=0.03)
    p.add_argument("--subtitle-bevel", type=float, default=0.008)
    p.add_argument("--gap", type=float, default=0.12,
                   help="vertical gap between title and subtitle, in text-curve units")
    p.add_argument("--target-height", type=float, default=0.045,
                   help="final total height (title+gap+subtitle) in meters")
    args = p.parse_args(argv)

    args.parsed_ranges = []
    for r in args.ranges:
        start, end, name = r.split(":")
        args.parsed_ranges.append((int(start), int(end), name))

    args.parsed_outs = {}
    for o in args.outs:
        name, path = o.split(":", 1)
        args.parsed_outs[name] = path

    return args


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_text(name, body, size, extrude, bevel, y_offset):
    bpy.ops.object.text_add(location=(0, y_offset, 0))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = body
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.extrude = extrude
    obj.data.bevel_depth = bevel
    bpy.context.view_layer.update()
    return obj


def split_by_color_groups(subtitle_obj, ranges):
    """Assign each subtitle character to a named material slot ("main" by
    default, or whatever `ranges` [(start,end,name), ...] overrides it to),
    convert to mesh, and separate by material. Returns {name: object}."""
    data = subtitle_obj.data
    name_by_char = ["main"] * len(data.body)
    for start, end, name in ranges:
        for i in range(start, min(end, len(name_by_char))):
            name_by_char[i] = name

    group_names = ["main"] + sorted({n for n in name_by_char if n != "main"})
    mat_index = {name: i for i, name in enumerate(group_names)}
    mats = {}
    for name in group_names:
        mat = bpy.data.materials.new(f"logo_{name}")
        data.materials.append(mat)
        mats[name] = mat

    for i, char_fmt in enumerate(data.body_format):
        char_fmt.material_index = mat_index[name_by_char[i]]

    bpy.ops.object.select_all(action="DESELECT")
    subtitle_obj.select_set(True)
    bpy.context.view_layer.objects.active = subtitle_obj
    bpy.ops.object.convert(target="MESH")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="MATERIAL")
    bpy.ops.object.mode_set(mode="OBJECT")

    pieces = list(bpy.context.selected_objects)
    result = {}
    for o in pieces:
        slot_mats = [s.material for s in o.material_slots if s.material]
        # a piece keeps only the faces of ONE material after separation, so
        # exactly one of our named materials will actually be present on it.
        matched = [name for name, mat in mats.items() if mat in slot_mats
                   and sum(1 for f in o.data.polygons if o.material_slots[f.material_index].material == mat) > 0]
        name = matched[0] if matched else "main"
        o.name = f"logo_subtitle_{name}"
        result[name] = o
    return result


def main():
    args = parse_args()
    clear_scene()

    if args.title_mesh:
        bpy.ops.wm.stl_import(filepath=args.title_mesh)
        title = bpy.context.selected_objects[0]
        title.name = "logo_title"
        title.location = (0, 0, 0)
        bpy.context.view_layer.update()
        # Rescale so its Y-extent (cap height) matches --title-size regardless
        # of whatever arbitrary units the source STL was traced/exported in --
        # only the RATIO to the subtitle's height matters here; the whole
        # group gets renormalized to --target-height at the very end anyway.
        rescale = args.title_size / title.dimensions.y
        title.scale = (rescale, rescale, rescale)
        bpy.ops.object.select_all(action="DESELECT")
        title.select_set(True)
        bpy.context.view_layer.objects.active = title
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        title_is_mesh = True
    else:
        title = make_text("logo_title", args.title, args.title_size,
                           args.title_extrude, args.title_bevel, 0.0)
        title_is_mesh = False
    subtitle = make_text("logo_subtitle", args.subtitle, args.subtitle_size,
                          args.subtitle_extrude, args.subtitle_bevel, 0.0)

    sub_y = -(title.dimensions.y / 2 + args.gap + subtitle.dimensions.y / 2)
    subtitle.location = (0, sub_y, 0)
    bpy.context.view_layer.update()

    import numpy as np
    import mathutils

    if args.parsed_ranges:
        sub_groups = split_by_color_groups(subtitle, args.parsed_ranges)
    else:
        bpy.ops.object.select_all(action="DESELECT")
        subtitle.select_set(True)
        bpy.context.view_layer.objects.active = subtitle
        bpy.ops.object.convert(target="MESH")
        sub_groups = {"main": subtitle}

    if not title_is_mesh:
        bpy.ops.object.select_all(action="DESELECT")
        title.select_set(True)
        bpy.context.view_layer.objects.active = title
        bpy.ops.object.convert(target="MESH")

    # Join title into the subtitle's "main" piece (title is always "main" color).
    sub_main = sub_groups["main"]
    bpy.ops.object.select_all(action="DESELECT")
    title.select_set(True)
    sub_main.select_set(True)
    bpy.context.view_layer.objects.active = sub_main
    bpy.ops.object.join()
    sub_groups["main"] = bpy.context.object
    sub_groups["main"].name = "logo_main"

    all_objs = list(sub_groups.values())

    # Rotate every piece identically (shared group orientation) so local +X
    # becomes the thin "outward normal" axis, +Y = width, +Z = height.
    for obj in all_objs:
        obj.rotation_euler = (1.5708, 0, 1.5708)
    bpy.context.view_layer.update()
    for obj in all_objs:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Measure the combined bbox across every piece so they all scale by the
    # exact same factor about the exact same pivot and stay aligned -- using
    # the bbox midpoint, NOT the mean of per-object centers, which would
    # skew the pivot when pieces have very different sizes.
    all_corners = []
    for obj in all_objs:
        bpy.context.view_layer.update()
        all_corners.extend(obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box)
    all_corners = np.array(all_corners)
    bbox_min = all_corners.min(axis=0)
    bbox_max = all_corners.max(axis=0)
    current_height = (bbox_max - bbox_min)[2]
    pivot = (bbox_min + bbox_max) / 2
    scale = args.target_height / current_height

    # Bake pivot-centered scaling directly into each object's vertex data
    # (rather than object transform + apply) so every piece ends up with
    # matrix_world == identity and therefore shares one exact coordinate frame.
    for obj in all_objs:
        mat = obj.matrix_world.copy()
        mesh = obj.data
        for v in mesh.vertices:
            world = mat @ v.co
            v.co = mathutils.Vector(((world[i] - pivot[i]) * scale for i in range(3)))
        obj.matrix_world = mathutils.Matrix.Identity(4)
        mesh.update()
    bpy.context.view_layer.update()

    for name, obj in sub_groups.items():
        corners = np.array([obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box])
        print(f"[generate_logo_mesh] '{name}' bbox size: {corners.max(axis=0) - corners.min(axis=0)}")
        out_path = args.parsed_outs.get(name)
        if not out_path:
            print(f"[generate_logo_mesh] no --out for '{name}', skipping export")
            continue
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.stl_export(filepath=out_path, export_selected_objects=True, global_scale=1.0)
        print(f"[generate_logo_mesh] exported '{name}' to {out_path}")


if __name__ == "__main__":
    main()
