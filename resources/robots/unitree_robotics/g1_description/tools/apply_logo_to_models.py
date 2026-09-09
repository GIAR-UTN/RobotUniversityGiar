"""Apply the GIAR chest/back emblem to every G1 URDF/MJCF variant, in place.

Idempotent: re-running after tweaking the constants below (position, color,
rotation, mesh filenames) safely replaces the previously-injected geoms
instead of duplicating them. This is the single source of truth for where
the logo sits -- never hand-edit a `logo_link`/`logo_back_link` block in an
individual .xml/.urdf file, edit the constants here and re-run instead.

Usage:
    .venv/bin/python resources/robots/unitree_robotics/g1_description/tools/apply_logo_to_models.py

Regenerating the mesh itself (new text, new split point, new sizing) is a
separate step -- see generate_logo_mesh.py's docstring -- run that first if
the wording/shape needs to change, then run this script to (re)place it.

Position tuning method (do this whenever MAIN_MESH's shape/size changes):
Render close-up crops via MuJoCo's Python API sweeping the depth-axis (local
X, relative to torso_link) offset until the emblem sits flush on the torso
surface at the target height -- too small = embedded inside the solid body
(invisible from every angle), too large = floats off the surface. See the
PR conversation / project memory `project_g1_giar_logo_swap.md` for the
empirical values already found for this torso mesh.
"""
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parent.parent  # .../g1_description


def offset(base, extra):
    return tuple(round(b + e, 7) for b, e in zip(base, extra))


# (mesh asset name, STL filename, rgba, URDF material name) for every colored
# piece of the emblem.
LOGO_PARTS = [
    ("logo_link", "giar_logo_link.STL", "0.06 0.28 0.55 1", "giar_blue"),          # title + "GIAR" in subtitle
    ("logo_accent", "giar_logo_link_accent.STL", "0.95 0.5 0.1 1", "giar_orange"), # "ru"
    ("logo_black", "giar_logo_link_black.STL", "0.05 0.05 0.05 1", "giar_black"),  # "stack"
]
# Kept for the URDF box-collision size and back-compat readability.
MAIN_MESH = LOGO_PARTS[0][1]
MAIN_RGBA = LOGO_PARTS[0][2]

FRONT_QUAT = "1 0 0 0"
BACK_QUAT = "0 0 0 1"              # 180 deg about Z -- reads correctly from behind

# The G1 description ships (at least) three differently-proportioned torso
# meshes across variants -- a placement tuned by render-sweep against one
# does NOT transfer to another. Each family below was tuned directly against
# its own mesh (see docstring for the render-sweep method); reuse a family's
# numbers only for a file confirmed to reference that exact STL.
TORSO_LINK_STL = ((0.10, 0, 0.16), (-0.085, 0, 0.16))                 # torso_link.STL
TORSO_LINK_REV_1_0_STL = ((0.10, 0, 0.16), (-0.085, 0, 0.16))         # torso_link_rev_1_0.STL (untuned: reusing TORSO_LINK_STL's numbers as a starting guess -- verify with a render sweep before trusting on a "_rev_1_0" 29dof/lock_waist/with_hand file)
TORSO_LINK_23DOF_REV_1_0_STL = ((0.0796, 0, 0.20), (-0.0686, 0, 0.20))  # torso_link_23dof_rev_1_0.STL -- tuned 2026-09-09 against g1_12dof.xml (the actual "g1" task model): flush-tuned to (0.075, -0.064), then nudged outward by the extra half-thickness from doubling the emblem's depth (giar_logo_link.STL main-piece X extent 0.0097->0.0190m) so the INNER face stays anchored at the same flush contact point and all the new thickness protrudes outward as a visibly raised plaque, per explicit request

# pelvis->torso_link fixed offset for g1_dual_arm.xml (worldbody-flat, torso
# geom itself sits here) -- it uses torso_link.STL, so add this offset to
# TORSO_LINK_STL's numbers to place it correctly.
DUAL_ARM_WAIST_OFFSET = (-0.0039635, 0, 0.054)

# Files whose XML has a distinct <body name="torso_link"> with the logo geom
# already carrying an explicit `pos` (any old value -- replaced unconditionally),
# mapped to which torso-mesh family (front_pos, back_pos) pair applies.
XML_TORSO_BODY_FILES = {
    "g1_23dof.xml": TORSO_LINK_STL,
    "g1_23dof_rev_1_0.xml": TORSO_LINK_23DOF_REV_1_0_STL,
    "g1_29dof.xml": TORSO_LINK_STL,
    "g1_29dof_rev_1_0.xml": TORSO_LINK_REV_1_0_STL,
    "g1_29dof_lock_waist.xml": TORSO_LINK_STL,
    "g1_29dof_lock_waist_rev_1_0.xml": TORSO_LINK_REV_1_0_STL,
    "g1_29dof_with_hand.xml": TORSO_LINK_STL,
    "g1_29dof_with_hand_rev_1_0.xml": TORSO_LINK_REV_1_0_STL,
}

# Files where the torso got merged into the pelvis body by the MJCF compiler
# (no separate torso_link body, so the logo geom needs its own explicit
# pelvis-local `pos`) -- (front_pos, back_pos) in pelvis-local coordinates.
XML_PELVIS_MERGED_ABSOLUTE = {
    "g1_12dof.xml": TORSO_LINK_23DOF_REV_1_0_STL,
    "g1_dual_arm.xml": (
        offset(DUAL_ARM_WAIST_OFFSET, TORSO_LINK_STL[0]),
        offset(DUAL_ARM_WAIST_OFFSET, TORSO_LINK_STL[1]),
    ),
}

# URDF link names are always "torso_link" regardless of which STL variant
# backs it -- map filename -> torso-mesh family the same way.
URDF_TORSO_FAMILY = {
    "g1_12dof.urdf": TORSO_LINK_23DOF_REV_1_0_STL,
    "g1_23dof.urdf": TORSO_LINK_STL,
    "g1_23dof_rev_1_0.urdf": TORSO_LINK_23DOF_REV_1_0_STL,
    "g1_29dof.urdf": TORSO_LINK_STL,
    "g1_29dof_rev_1_0.urdf": TORSO_LINK_REV_1_0_STL,
    "g1_29dof_lock_waist.urdf": TORSO_LINK_STL,
    "g1_29dof_lock_waist_rev_1_0.urdf": TORSO_LINK_REV_1_0_STL,
    "g1_29dof_with_hand.urdf": TORSO_LINK_STL,
    "g1_29dof_with_hand_rev_1_0.urdf": TORSO_LINK_REV_1_0_STL,
    "g1_dual_arm.urdf": TORSO_LINK_STL,
}

START_MARK = "<!-- GIAR LOGO (generated by apply_logo_to_models.py; do not hand-edit) -->"
END_MARK = "<!-- /GIAR LOGO -->"


def fmt(pos):
    return " ".join(f"{v:g}" for v in pos)


def xml_mesh_assets_block():
    return "\n".join(f'    <mesh name="{mesh_name}" file="{fname}"/>'
                      for mesh_name, fname, _, _ in LOGO_PARTS)


def xml_material_assets_block():
    # specular=0 shininess=0 reflectance=0 -> flat matte color, no glossy
    # highlight riding the extrusion's side walls (which otherwise reads as
    # a fake bevel/drop-shadow, like the piece has more 3D relief than it
    # actually does).
    return "\n".join(f'    <material name="{mesh_name}_mat" rgba="{rgba}" '
                      f'specular="0" shininess="0" reflectance="0"/>'
                      for mesh_name, _, rgba, _ in LOGO_PARTS)


def xml_geom_block(indent, front_pos, back_pos):
    # No physics collision geom on purpose: this is a cosmetic decal welded
    # to torso_link via a fixed joint. Genesis does not appear to auto-exclude
    # parent/fixed-child collision the way MuJoCo's contype/conaffinity=0
    # convention would suggest, so a colliding proxy here gets pushed off the
    # body by the solver over time -- looks like the logo "floating away"
    # from the chest after the sim has run a while. contype=0 conaffinity=0
    # on every geom below means it never participates in collision at all.
    ind = " " * indent
    lines = [START_MARK]
    for pos, quat, tag in [(front_pos, FRONT_QUAT, "front"), (back_pos, BACK_QUAT, "back")]:
        p = fmt(pos)
        for mesh_name, _, _, _ in LOGO_PARTS:
            lines.append(f'{ind}<geom pos="{p}" quat="{quat}" type="mesh" contype="0" conaffinity="0" '
                         f'group="1" density="0" material="{mesh_name}_mat" mesh="{mesh_name}"/>')
    lines.append(END_MARK)
    return ("\n" + ind).join(lines)


def replace_generated_block(text, new_block, indent):
    """Remove any previous GIAR LOGO block (idempotency) and return text with
    a marker for insert_after_marker to place the new one."""
    pattern = re.compile(
        re.escape(START_MARK) + r".*?" + re.escape(END_MARK), re.DOTALL)
    if pattern.search(text):
        return pattern.sub(new_block.strip("\n"), text, count=1)
    return None  # caller must insert fresh


def ensure_mesh_assets(text, fname):
    """Make sure exactly one <mesh> entry exists per LOGO_PARTS name, in one
    place (right where logo_link's asset line already was), removing any
    stale entries from a previous run with a different part list."""
    if '<mesh name="logo_link" file="' not in text:
        raise RuntimeError(f"{fname}: couldn't find logo_link mesh asset line")
    for mesh_name, _, _, _ in LOGO_PARTS[1:]:
        text = re.sub(rf'    <mesh name="{mesh_name}" file="[^"]*"/>\n?', "", text)
    for mesh_name, _, _, _ in LOGO_PARTS:
        text = re.sub(rf'    <material name="{mesh_name}_mat"[^/]*/>\n?', "", text)
    text = re.sub(r'<mesh name="logo_link" file="[^"]*"/>',
                   xml_mesh_assets_block() + "\n" + xml_material_assets_block(), text, count=1)
    return text


def update_xml_torso_body(fname):
    path = BASE / fname
    text = path.read_text()
    text = ensure_mesh_assets(text, fname)

    front_pos, back_pos = XML_TORSO_BODY_FILES[fname]
    indent = 12
    new_block = xml_geom_block(indent, front_pos, back_pos)

    replaced = replace_generated_block(text, new_block, indent)
    if replaced is not None:
        text = replaced
    else:
        # first run on this file: remove the old (pre-GIAR-script) 2-line
        # logo_link geom pair and insert our marked block in its place.
        old_pair = re.compile(
            r' *<geom[^>]*mesh="logo_link"/>\n(?: *<geom[^>]*mesh="logo_link"[^>]*/>\n?)*')
        if not old_pair.search(text):
            raise RuntimeError(f"{fname}: couldn't find original logo_link geom(s) to replace")
        text = old_pair.sub(new_block + "\n", text, count=1)

    path.write_text(text)
    print(f"updated {fname}")


def update_xml_pelvis_merged(fname, front_pos, back_pos):
    path = BASE / fname
    text = path.read_text()
    text = ensure_mesh_assets(text, fname)

    indent = 6
    new_block = xml_geom_block(indent, front_pos, back_pos)

    replaced = replace_generated_block(text, new_block, indent)
    if replaced is not None:
        text = replaced
    else:
        old_pair = re.compile(
            r' *<geom[^>]*mesh="logo_link"/>\n(?: *<geom[^>]*mesh="logo_link"[^>]*/>\n?)*')
        if not old_pair.search(text):
            raise RuntimeError(f"{fname}: couldn't find original logo_link geom(s) to replace")
        text = old_pair.sub(new_block + "\n", text, count=1)

    path.write_text(text)
    print(f"updated {fname}")


def urdf_link_block(link_name, joint_name, joint_pos, joint_rpy):
    # No <collision> element on purpose -- see the comment in xml_geom_block:
    # a colliding proxy on a fixed-jointed cosmetic decal gets pushed off the
    # body by the physics solver over time instead of staying welded flush.
    visuals = "\n".join(f"""    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/{fname}"/>
      </geometry>
      <material name="{mat_name}">
        <color rgba="{rgba}"/>
      </material>
    </visual>""" for mesh_name, fname, rgba, mat_name in LOGO_PARTS)
    return f"""  <joint name="{joint_name}" type="fixed">
    <origin xyz="{fmt(joint_pos)}" rpy="{joint_rpy}"/>
    <parent link="torso_link"/>
    <child link="{link_name}"/>
  </joint>
  <link name="{link_name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.001"/>
      <inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-7" iyz="0" izz="1e-7"/>
    </inertial>
{visuals}
  </link>"""


def urdf_logo_block(front_pos, back_pos):
    front = urdf_link_block("logo_link", "logo_joint", front_pos, "0 0 0")
    back = urdf_link_block("logo_back_link", "logo_back_joint", back_pos, "0 0 3.14159265")
    return f"  {START_MARK}\n{front}\n\n{back}\n  {END_MARK}"


def update_urdf(fname, front_pos, back_pos):
    path = BASE / fname
    # These ship with CRLF line endings; read_text() silently normalizes to
    # \n (universal newlines), so writing back must restore \r\n explicitly
    # or every line in the file shows as changed in `git diff`.
    raw = path.read_bytes()
    uses_crlf = b"\r\n" in raw
    text = raw.decode().replace("\r\n", "\n")

    pattern = re.compile(
        re.escape(START_MARK) + r".*?" + re.escape(END_MARK), re.DOTALL)
    new_block = urdf_logo_block(front_pos, back_pos)

    if pattern.search(text):
        text = pattern.sub(new_block, text, count=1)
    else:
        old_block = re.compile(r'  <!-- LOGO -->.*?</link>\n', re.DOTALL)
        if not old_block.search(text):
            raise RuntimeError(f"{fname}: couldn't find original LOGO block to replace")
        text = old_block.sub(new_block + "\n", text, count=1)

    if uses_crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode())
    print(f"updated {fname}")


def main():
    for fname in XML_TORSO_BODY_FILES:
        update_xml_torso_body(fname)
    for fname, (front_pos, back_pos) in XML_PELVIS_MERGED_ABSOLUTE.items():
        update_xml_pelvis_merged(fname, front_pos, back_pos)
    for fname, (front_pos, back_pos) in URDF_TORSO_FAMILY.items():
        update_urdf(fname, front_pos, back_pos)
    print("done")


if __name__ == "__main__":
    main()
