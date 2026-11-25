"""Prototype A: Blender 3D toon character + CC0 mocap-style animation library + AI-painted 2D plate.

blender -b --factory-startup -P nell_beat.py -- <ual_dir> <bg_plate.png> <out_dir> [engine EEVEE|CYCLES] [res_scale]

Everything is built by code (what Claude would generate per shot from a JSON timeline):
  - character: Quaternius Female Mannequin (CC0), dressed procedurally (cardigan/skin/shoe materials by bone
    weight, a skinned A-line skirt, hair cap + bun, glasses, eyes)
  - motion: Quaternius Universal Animation Library 1+2 (CC0) clips sequenced on the NLA with blends
    (Walk_Formal_Loop -> Farm_PlantSeed kneel -> custom kneel+carry pose built from Walk_Carry_Loop arms)
  - squirrel: primitives, attached to the right hand with an animated Child-Of constraint at the contact frame
  - toon look: Shader-to-RGB 3-band ramp + inverted-hull outline
  - scene: AI-painted plate (Z-Image-Turbo, Apache-2.0) far behind, 3D falling maple leaves in front/behind,
    always-moving camera (push-in + drift + tilt)
"""
import bpy, bmesh, math, random, sys, time
from mathutils import Vector, Matrix, Quaternion
from bpy_extras.object_utils import world_to_camera_view

argv = sys.argv[sys.argv.index("--") + 1:]
U, PLATE, OUT = argv[0].rstrip("/") + "/", argv[1], argv[2].rstrip("/") + "/"
ENGINE = argv[3] if len(argv) > 3 else "EEVEE"
RS = float(argv[4]) if len(argv) > 4 else 1.0
FPS, N = 30, 270
random.seed(4)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = FPS; sc.frame_start = 1; sc.frame_end = N

# ---------------- assets ----------------
bpy.ops.import_scene.gltf(filepath=U + "Universal Animation Library 2 [Standard]/Female Mannequin/Unreal-Godot/Mannequin_F.glb")
ARM = bpy.data.objects["Armature"]; BODY = bpy.data.objects["Mannequin_F"]
keep = {"Armature", "Mannequin_F"}
def purge():
    for o in list(sc.objects):
        if o.name not in keep: bpy.data.objects.remove(o)
purge()
bpy.ops.import_scene.fbx(filepath=U + "Animation Library[Standard]/Unreal Engine/AL_Standard.fbx"); purge()
bpy.ops.import_scene.gltf(filepath=U + "Universal Animation Library 2 [Standard]/Unreal-Godot/UAL2_Standard.glb"); purge()
A = {k.split("|")[-1]: v for k, v in bpy.data.actions.items()}

# ---------------- toon materials ----------------
def toon(name, rgb, shadow=0.55, outline=False):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if outline:
        em = nt.nodes.new("ShaderNodeEmission"); em.inputs[0].default_value = (*rgb, 1); em.inputs[1].default_value = 1
        nt.links.new(em.outputs[0], out.inputs[0]); m.use_backface_culling = True
        return m
    d = nt.nodes.new("ShaderNodeBsdfDiffuse"); s2r = nt.nodes.new("ShaderNodeShaderToRGB")
    ramp = nt.nodes.new("ShaderNodeValToRGB"); ramp.color_ramp.interpolation = "CONSTANT"
    e = ramp.color_ramp.elements; e[0].position = 0.0; e[0].color = (shadow,) * 3 + (1,)
    e[1].position = 0.22; e[1].color = (0.82,) * 3 + (1,)
    e3 = e.new(0.55); e3.color = (1, 1, 1, 1)
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"; mix.inputs[0].default_value = 1
    mix.inputs[6].default_value = (*rgb, 1)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(d.outputs[0], s2r.inputs[0]); nt.links.new(s2r.outputs[0], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], mix.inputs[7]); nt.links.new(mix.outputs[2], em.inputs[0]); nt.links.new(em.outputs[0], out.inputs[0])
    return m

C = dict(skin=(0.93, 0.74, 0.62), card=(0.12, 0.34, 0.20), blouse=(0.95, 0.90, 0.76), skirt=(0.55, 0.22, 0.13),
         shoe=(0.30, 0.17, 0.10), hair=(0.86, 0.86, 0.88), dark=(0.10, 0.07, 0.06), sq=(0.42, 0.39, 0.38), sqb=(0.85, 0.80, 0.72))
M = {k: toon(k, v) for k, v in C.items()}
M["line"] = toon("line", (0.16, 0.09, 0.06), outline=True)

def add_outline(o, th=0.006):
    o.data.materials.append(M["line"])
    s = o.modifiers.new("outline", "SOLIDIFY"); s.thickness = -th; s.use_flip_normals = True
    s.material_offset = len(o.data.materials) - 1; s.use_rim = False

# body materials from dominant bone weight
BODY.data.materials.clear()
for k in ["skin", "card", "blouse", "skirt", "shoe"]: BODY.data.materials.append(M[k])
idx = {k: i for i, k in enumerate(["skin", "card", "blouse", "skirt", "shoe"])}
gname = {g.index: g.name for g in BODY.vertex_groups}
def region(bn, z, x=0.0, y=0.0):
    if bn.startswith(("Head", "neck")): return "skin"
    if any(bn.startswith(p) for p in ("hand", "thumb", "index", "middle", "ring", "pinky")): return "skin"
    if bn.startswith(("foot", "ball")): return "shoe"
    if bn.startswith(("thigh", "calf", "pelvis")): return "skirt"
    if bn in ("spine_03", "neck_01") and z > 1.36 and y < -0.02 and abs(x) < 0.075: return "blouse"
    return "card"
me = BODY.data
for p in me.polygons:
    votes = {}
    for vi in p.vertices:
        v = me.vertices[vi]
        if not v.groups: continue
        g = max(v.groups, key=lambda g: g.weight)
        wv = BODY.matrix_world @ v.co
        r = region(gname[g.group], wv.z, wv.x, wv.y)
        votes[r] = votes.get(r, 0) + 1
    p.material_index = idx[max(votes, key=votes.get)] if votes else idx["card"]
    p.use_smooth = True
add_outline(BODY, 0.007)

# rest-pose measurements
ARM.data.pose_position = "REST"; bpy.context.view_layer.update()
def bone_w(name, tail=False):
    b = ARM.data.bones[name]; return ARM.matrix_world @ (b.tail_local if tail else b.head_local)
head_vs = [BODY.matrix_world @ v.co for v in me.vertices if v.groups and gname[max(v.groups, key=lambda g: g.weight).group] == "Head"]
hmin = Vector((min(v.x for v in head_vs), min(v.y for v in head_vs), min(v.z for v in head_vs)))
hmax = Vector((max(v.x for v in head_vs), max(v.y for v in head_vs), max(v.z for v in head_vs)))
hc = (hmin + hmax) / 2; hs = hmax - hmin
pelvis_z = bone_w("pelvis").z
print("HEAD", hc, hs, "PELVIS", pelvis_z)

def parent_bone(o, bone):
    pb = ARM.pose.bones[bone]
    o.parent = ARM; o.parent_type = "BONE"; o.parent_bone = bone
    o.matrix_parent_inverse = (ARM.matrix_world @ pb.matrix @ Matrix.Translation((0, pb.bone.length, 0))).inverted()

def sphere(name, loc, scale, mat, seg=24, outline=0.004):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=seg // 2, location=loc)
    o = bpy.context.object; o.name = name; o.scale = scale; o.data.materials.append(M[mat])
    for p in o.data.polygons: p.use_smooth = True
    bpy.ops.object.transform_apply(scale=True)
    if outline: add_outline(o, outline)
    return o

# hair: cap over the skull + low bun at the back (character faces -Y in rest)
cap = sphere("hair_cap", hc + Vector((0, 0.03, 0.045)), (hs.x * 0.56, hs.y * 0.5, hs.z * 0.4), "hair")
bun = sphere("hair_bun", hc + Vector((0, hs.y * 0.55, -hs.z * 0.12)), (0.05, 0.045, 0.045), "hair")
# eyes + glasses on the face (-Y side)
fy = hmin.y - 0.002
eyes = [sphere(f"eye{s}", Vector((hc.x + s * 0.032, fy + 0.012, hc.z + 0.004)), (0.009, 0.006, 0.012), "dark", 12, 0) for s in (-1, 1)]
glasses = []
for s in (-1, 1):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.024, minor_radius=0.0028, location=(hc.x + s * 0.032, fy - 0.004, hc.z + 0.004),
                                     rotation=(math.pi / 2, 0, 0))
    g = bpy.context.object; g.data.materials.append(M["dark"]); glasses.append(g)
bpy.ops.mesh.primitive_cylinder_add(radius=0.0025, depth=0.018, location=(hc.x, fy - 0.004, hc.z + 0.01), rotation=(0, math.pi / 2, 0))
glasses.append(bpy.context.object); glasses[-1].data.materials.append(M["dark"])
mouth = sphere("mouth", Vector((hc.x, fy + 0.006, hc.z - 0.045)), (0.012, 0.004, 0.003), "dark", 12, 0)
for o in [cap, bun, *eyes, *glasses, mouth]: parent_bone(o, "Head")

# skinned A-line skirt: waist ring weighted to pelvis, hem rings blended toward the thigh on that side
wz, hz = pelvis_z + 0.06, 0.16
bm = bmesh.new(); rings, segs = 9, 40
px, py = bone_w("pelvis").x, bone_w("pelvis").y
verts = []
for r in range(rings):
    t = r / (rings - 1); z = wz + (hz - wz) * t; rad = 0.15 + 0.11 * t ** 0.9
    ring = []
    for s in range(segs):
        a = 2 * math.pi * s / segs
        ring.append(bm.verts.new((px + rad * math.cos(a), py + rad * 0.9 * math.sin(a), z)))
    verts.append(ring)
for r in range(rings - 1):
    for s in range(segs):
        bm.faces.new((verts[r][s], verts[r][(s + 1) % segs], verts[r + 1][(s + 1) % segs], verts[r + 1][s]))
skm = bpy.data.meshes.new("skirt"); bm.to_mesh(skm); bm.free()
SK = bpy.data.objects.new("skirt", skm); sc.collection.objects.link(SK)
for p in skm.polygons: p.use_smooth = True
skm.materials.append(M["skirt"])
gp, gl, gr = (SK.vertex_groups.new(name=n) for n in ("pelvis", "thigh_l", "thigh_r"))
xl = bone_w("thigh_l").x
for v in skm.vertices:
    t = max(0.0, min(1.0, (wz - v.co.z) / (wz - hz)))
    side = (v.co.x - px) / 0.3 * (1 if xl > px else -1)   # +1 toward left thigh
    leg = 0.8 * t ** 1.1                                    # never fully follows one leg -> no trouser split
    wl = leg * max(0.0, min(1.0, 0.5 + side)); wr = leg * max(0.0, min(1.0, 0.5 - side))
    gp.add([v.index], 1.0 - wl - wr, "REPLACE"); gl.add([v.index], wl, "REPLACE"); gr.add([v.index], wr, "REPLACE")
SK.parent = ARM; m = SK.modifiers.new("arm", "ARMATURE"); m.object = ARM
sm = SK.modifiers.new("smooth", "CORRECTIVE_SMOOTH"); sm.iterations = 6
sd = SK.modifiers.new("solid", "SOLIDIFY"); sd.thickness = 0.006
add_outline(SK, 0.006)

ARM.data.pose_position = "POSE"
for o in [ARM]:
    o.rotation_mode = "XYZ"; o.rotation_euler = (0, 0, math.radians(90))   # face +X (toward the tree)

# ---------------- animation: NLA timeline ----------------
ARM.animation_data_create()
def strip(track_name, act, start, a0, a1, scale=1.0, bin_=0, bout=0, repeat=1.0, extrap="NOTHING", blend="REPLACE"):
    tr = ARM.animation_data.nla_tracks.new(); tr.name = track_name
    s = tr.strips.new(track_name, int(start), act)
    s.action_frame_start = a0; s.action_frame_end = a1; s.scale = scale; s.repeat = repeat
    s.blend_in = bin_; s.blend_out = bout; s.extrapolation = extrap; s.blend_type = blend
    s.use_auto_blend = False
    return s

# custom pose action: kneeling legs (PlantSeed @ kneel) + carrying arms (Walk_Carry_Loop @ 1)
UPPER = ("clavicle", "upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky")
def pose_from(act, frame):
    ARM.animation_data.action = act
    try: ARM.animation_data.action_slot = act.slots[0]
    except Exception: pass
    sc.frame_set(frame)
    return {pb.name: (pb.location.copy(), pb.rotation_quaternion.copy()) for pb in ARM.pose.bones}
# frame in Farm_PlantSeed where the right hand is lowest = the "touch the ground" moment
ARM.animation_data.action = A["Farm_PlantSeed"]
try: ARM.animation_data.action_slot = A["Farm_PlantSeed"].slots[0]
except Exception: pass
hz_ = []
for f in range(1, 84):
    sc.frame_set(f); hz_.append((ARM.pose.bones["hand_r"].head.z, f))
KNEEL_F = min(hz_)[1]; print("KNEEL_F", KNEEL_F, min(hz_))
kneel = pose_from(A["Farm_PlantSeed"], KNEEL_F)
carry = pose_from(A["Walk_Carry_Loop"], 1)
ARM.animation_data.action = None
kc = bpy.data.actions.new("KneelCarry")
ARM.animation_data.action = kc
for pb in ARM.pose.bones:
    loc, rot = carry[pb.name] if pb.name.startswith(UPPER) else kneel[pb.name]
    pb.location = loc; pb.rotation_quaternion = rot
    pb.keyframe_insert("location", frame=1); pb.keyframe_insert("rotation_quaternion", frame=1)
# head look down at the squirrel then up with a small smile-tilt (additive layer)
hd = bpy.data.actions.new("HeadAct"); ARM.animation_data.action = hd
hb = ARM.pose.bones["Head"]
for f, ang, roll in [(1, 0, 0), (150, 0.22, 0.0), (205, 0.12, 0.10), (235, 0.05, -0.06), (270, 0.1, 0.04)]:
    hb.rotation_quaternion = Quaternion((1, 0, 0), ang) @ Quaternion((0, 0, 1), roll)
    hb.keyframe_insert("rotation_quaternion", frame=f)
ARM.animation_data.action = None

WALK_END = 96
strip("walk", A["Walk_Formal_Loop"], 1, 1, 41, scale=1.12, repeat=WALK_END / (40 * 1.12), bout=10)
KS = 1.5
strip("kneel", A["Farm_PlantSeed"], WALK_END - 8, 1, KNEEL_F, scale=KS, bin_=12, extrap="HOLD_FORWARD")
KNEEL_DONE = int(WALK_END - 8 + KNEEL_F * KS)
strip("carry", kc, KNEEL_DONE + 14, 1, 2, scale=1.0, bin_=28, extrap="HOLD_FORWARD")
print("KNEEL_DONE", KNEEL_DONE)
strip("head", hd, 1, 1, 270, blend="COMBINE")

# ground positions from the plate: solve where screen points hit the ground plane z=0
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 40; cam.data.sensor_fit = "VERTICAL"; cam.data.sensor_height = 24
sc.render.resolution_x, sc.render.resolution_y = int(720 * RS), int(1280 * RS)
cam.location = (0, -9.0, 2.2); cam.rotation_euler = (math.radians(87), 0, 0)
bpy.context.view_layer.update()
def ground_at(sx, sy):
    """screen (0..1, 0..1 bottom-up) -> world point on z=0"""
    cd = cam.data; h = cd.sensor_height; w = h * sc.render.resolution_x / sc.render.resolution_y
    d = Vector(((sx - 0.5) * w, (sy - 0.5) * h, -cd.lens))
    d = cam.matrix_world.to_3x3() @ d; o = cam.matrix_world.translation
    t = -o.z / d.z; return o + d * t
P0 = ground_at(0.04, 0.14); P1 = ground_at(0.50, 0.14)
SQ_POS = ground_at(0.66, 0.14)
print("P0", P0, "P1", P1, "SQ", SQ_POS)
# root motion along P0->P1 (ease-out), facing direction of travel then turn toward the squirrel
dirv = (P1 - P0); yaw = math.atan2(dirv.y, dirv.x)
for f in range(1, WALK_END + 9, 3):
    t = min(1.0, (f - 1) / (WALK_END - 1)); e = 1 - (1 - t) ** 1.6 if t > 0.75 else t * 0.96 / 0.75 * 0.75 / 0.96
    ARM.location = P0.lerp(P1, min(1.0, t)); ARM.keyframe_insert("location", frame=f)
for f, yw in [(1, yaw), (WALK_END - 10, yaw), (WALK_END + 14, math.atan2(SQ_POS.y - P1.y, SQ_POS.x - P1.x))]:
    ARM.rotation_euler = (0, 0, yw + math.radians(90)); ARM.keyframe_insert("rotation_euler", frame=f)
ARM.rotation_euler = (0, 0, 0)

# ---------------- squirrel ----------------
sq_parts = []
body = sphere("sq_body", (0, 0, 0.07), (0.05, 0.04, 0.065), "sq"); sq_parts.append(body)
belly = sphere("sq_belly", (-0.03, 0, 0.065), (0.025, 0.03, 0.05), "sqb", 16, 0); sq_parts.append(belly)
head = sphere("sq_head", (-0.02, 0, 0.15), (0.035, 0.032, 0.033), "sq"); sq_parts.append(head)
for s in (-1, 1):
    sq_parts.append(sphere(f"sq_ear{s}", (-0.005, s * 0.02, 0.185), (0.008, 0.006, 0.016), "sq", 12, 0.002))
    sq_parts.append(sphere(f"sq_eye{s}", (-0.048, s * 0.016, 0.158), (0.006, 0.006, 0.007), "dark", 10, 0))
tail = [sphere(f"sq_tail{i}", (0.05 + 0.035 * math.sin(i * 0.5), 0, 0.03 + i * 0.035), (0.03 + 0.01 * i / 5, 0.03, 0.03), "sq", 16, 0.003)
        for i in range(6)]
SQ = bpy.data.objects.new("squirrel", None); sc.collection.objects.link(SQ)
for o in sq_parts + tail: o.parent = SQ
SQ.location = SQ_POS; SQ.rotation_euler = (0, 0, math.atan2(P1.y - SQ_POS.y, P1.x - SQ_POS.x) + math.pi)
for f in range(1, N + 1, 6):   # tail flicks
    for i, t in enumerate(tail):
        t.rotation_euler = (0.18 * math.sin(f * 0.21 + i * 0.6), 0, 0); t.keyframe_insert("rotation_euler", frame=f)

# find the contact frame: right hand lowest point during the kneel, move squirrel under it
best = (9e9, 120)
for f in range(KNEEL_DONE - 20, KNEEL_DONE + 14, 1):
    sc.frame_set(f); hw = ARM.matrix_world @ ARM.pose.bones["hand_r"].head
    if hw.z < best[0]: best = (hw.z, f, hw.copy())
CONTACT = best[1]; hw = best[2]
SQ.location = (hw.x + 0.18, hw.y, 0.0); SQ.keyframe_insert("location", frame=1); SQ.keyframe_insert("location", frame=CONTACT - 14)
SQ.location = (hw.x + 0.08, hw.y, hw.z * 0.5 + 0.18); SQ.keyframe_insert("location", frame=CONTACT - 7)
SQ.location = (hw.x, hw.y, hw.z - 0.06); SQ.keyframe_insert("location", frame=CONTACT)
print("CONTACT", CONTACT, hw)
sc.frame_set(CONTACT)
co = SQ.constraints.new("CHILD_OF"); co.target = ARM; co.subtarget = "hand_r"
co.influence = 0; co.keyframe_insert("influence", frame=CONTACT - 1)
bpy.context.view_layer.update()
pm = ARM.matrix_world @ ARM.pose.bones["hand_r"].matrix
co.inverse_matrix = pm.inverted()
co.influence = 1; co.keyframe_insert("influence", frame=CONTACT)

# ---------------- plate + leaves + camera move ----------------
img = bpy.data.images.load(PLATE)
DIST = 26.0
bpy.ops.mesh.primitive_plane_add()
plate = bpy.context.object; plate.name = "plate"
ph = 2 * DIST * math.tan(math.atan(12 / 40)) * 1.18; pw = ph * img.size[0] / img.size[1]
plate.scale = (pw / 2, ph / 2, 1); plate.rotation_euler = cam.rotation_euler
plate.location = cam.location + cam.matrix_world.to_3x3() @ Vector((0, -0.6, -DIST))
pm_ = bpy.data.materials.new("plate"); pm_.use_nodes = True; nt = pm_.node_tree; nt.nodes.clear()
tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img; em = nt.nodes.new("ShaderNodeEmission"); o_ = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(tex.outputs[0], em.inputs[0]); nt.links.new(em.outputs[0], o_.inputs[0]); plate.data.materials.append(pm_)
# NOTE: ground in the plate sits where the character's z=0 projects only from the start camera; plate is far
# so camera drift gives real parallax between character, leaves and painted background.

leaf_cols = [(0.86, 0.30, 0.08), (0.93, 0.55, 0.12), (0.72, 0.16, 0.07), (0.95, 0.70, 0.25)]
leafm = [toon(f"leaf{i}", c, 0.75) for i, c in enumerate(leaf_cols)]
for m_ in leafm: m_.use_backface_culling = False
def leaf_mesh():
    bm = bmesh.new(); pts = []
    for k in range(10):
        a = 2 * math.pi * k / 10; r = (1.0 if k % 2 == 0 else 0.62) * (1.25 if k == 0 else 1.0)
        pts.append(bm.verts.new((r * math.cos(a), r * math.sin(a), 0)))
    bm.faces.new(pts); m_ = bpy.data.meshes.new("leaf"); bm.to_mesh(m_); bm.free(); return m_
lm = leaf_mesh()
for i in range(70):
    o = bpy.data.objects.new(f"leaf{i}", lm.copy()); sc.collection.objects.link(o); o.data.materials.append(random.choice(leafm))
    depth = random.uniform(-6.5, 9.0)   # some in front of Nell (foreground parallax), some behind
    x0 = random.uniform(-4.5, 5.5) + depth * 0.25; z0 = random.uniform(0.5, 6.0)
    s = random.uniform(0.03, 0.05) * (1.3 if depth < -3 else 1.0); o.scale = (s, s * 0.9, s)
    sp = random.uniform(0.35, 0.7); ph_ = random.uniform(0, 6.28); dr = random.uniform(-0.25, 0.1)
    for f in range(1, N + 1, 5):
        tt = f / FPS; z = z0 - sp * tt
        z = (z % 6.2) - 0.2
        o.location = (x0 + 0.35 * math.sin(tt * 1.7 + ph_) + dr * tt, depth, max(0.02, z))
        o.rotation_euler = (1.2 * math.sin(tt * 2.3 + ph_), 0.9 * math.cos(tt * 1.9 + ph_), tt * 1.5 + ph_)
        o.keyframe_insert("location", frame=f); o.keyframe_insert("rotation_euler", frame=f)

# camera: slow push-in + drift right + gentle tilt (always moving), eased
for f, dx, dy, lens, tilt in [(1, -0.35, -9.0, 40, 87), (90, 0.15, -8.7, 41.5, 86.8), (170, 0.55, -8.3, 44, 86.4), (270, 0.85, -8.0, 47, 86.6)]:
    cam.location = (dx, dy, 2.2); cam.data.lens = lens; cam.rotation_euler = (math.radians(tilt), 0, math.radians(1.2 * dx))
    cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_euler", frame=f); cam.data.keyframe_insert("lens", frame=f)
# the plate is parented to the camera-independent world; scale it a bit so edges never show
plate.scale = (plate.scale[0] * 1.15, plate.scale[1] * 1.15, 1)

# soft contact shadow under Nell
bpy.ops.mesh.primitive_circle_add(vertices=32, radius=0.32, fill_type="NGON")
sh = bpy.context.object; sh.scale = (1.0, 0.55, 1); sh.location.z = 0.005
shm = bpy.data.materials.new("shadow"); shm.use_nodes = True; nt = shm.node_tree; nt.nodes.clear()
tr_ = nt.nodes.new("ShaderNodeBsdfTransparent"); em_ = nt.nodes.new("ShaderNodeEmission"); em_.inputs[0].default_value = (0.25, 0.2, 0.08, 1)
mx_ = nt.nodes.new("ShaderNodeMixShader"); mx_.inputs[0].default_value = 0.3
o_ = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(tr_.outputs[0], mx_.inputs[1]); nt.links.new(em_.outputs[0], mx_.inputs[2])
nt.links.new(mx_.outputs[0], o_.inputs[0]); sh.data.materials.append(shm)
cl = sh.constraints.new("COPY_LOCATION"); cl.target = ARM; cl.subtarget = "pelvis"; cl.use_z = False

# lights / world / render
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
sun.rotation_euler = (math.radians(50), math.radians(-25), math.radians(-35)); sun.data.energy = 3
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.75, 0.55, 1); w.node_tree.nodes["Background"].inputs[1].default_value = 0.6
sc.view_settings.view_transform = "Standard"
if ENGINE == "CYCLES":
    sc.render.engine = "CYCLES"; sc.cycles.samples = 16; sc.cycles.use_denoising = True; sc.cycles.device = "CPU"
else:
    try: sc.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError: sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = 16
sc.render.film_transparent = False
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = OUT + "frames/f_"
bpy.ops.wm.save_as_mainfile(filepath=OUT + "nell_beat.blend")
t0 = time.time()
if len(argv) > 5 and argv[5] == "POSE":   # export 2D OpenPose-18 joints per frame (control signal for AI repaint)
    import json
    def p2(v):
        c = world_to_camera_view(sc, cam, v); return [round(c.x, 4), round(1 - c.y, 4)]
    def bw(n, tail=False):
        pb = ARM.pose.bones[n]; return ARM.matrix_world @ (pb.tail if tail else pb.head)
    frames = []
    for f in range(1, N + 1):
        sc.frame_set(f)
        hm = ARM.matrix_world @ ARM.pose.bones["Head"].matrix
        hcen = hm @ Vector((0, 0.09, 0)); fwd = (hm.to_3x3() @ Vector((0, 0, 1))).normalized()
        # pick the head axis that points most toward the face: try +Z/-Z of the bone and use the one closest to body facing
        face = (ARM.matrix_world.to_3x3() @ Vector((0, -1, 0))).normalized()
        cand = [hm.to_3x3() @ Vector(v) for v in [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0)]]
        fwd = max((c_.normalized() for c_ in cand), key=lambda c_: c_.dot(face))
        side = fwd.cross(Vector((0, 0, 1))).normalized()
        nose = hcen + fwd * 0.10
        J = [nose, bw("neck_01"), bw("upperarm_r"), bw("lowerarm_r"), bw("hand_r"), bw("upperarm_l"), bw("lowerarm_l"), bw("hand_l"),
             bw("thigh_r"), bw("calf_r"), bw("foot_r"), bw("thigh_l"), bw("calf_l"), bw("foot_l"),
             hcen + fwd * 0.08 - side * 0.035, hcen + fwd * 0.08 + side * 0.035, hcen - side * 0.075, hcen + side * 0.075]
        frames.append({"f": f, "j": [p2(v) for v in J], "sq": p2(SQ.matrix_world.translation + Vector((0, 0, 0.1)))})
    open(OUT + "pose2d.json", "w").write(json.dumps({"fps": FPS, "w": sc.render.resolution_x, "h": sc.render.resolution_y, "frames": frames}))
    print("POSE_EXPORTED", len(frames)); sys.exit(0)
if len(argv) > 5:   # stills mode for quick checks
    for f in map(int, argv[5].split(",")):
        sc.frame_set(f); sc.render.filepath = OUT + f"still_{f:03d}.png"; bpy.ops.render.render(write_still=True)
    print("STILLS_S", round(time.time() - t0, 1)); sys.exit(0)
bpy.ops.render.render(animation=True)
print("RENDER_S", round(time.time() - t0, 1), "FRAMES", N, "ENGINE", sc.render.engine, "RES", sc.render.resolution_x, sc.render.resolution_y)
