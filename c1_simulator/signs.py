"""Put readable signs on walls in an Isaac scene.

Two halves:
  make_sign_texture()  — draw "CAFETERIA <-" into a PNG (pure PIL, no Isaac needed)
  add_sign_to_stage()  — hang that PNG on a quad in the USD stage at a world pose

Why a real textured quad rather than a coloured box: the detector and the VLM have to actually
SEE a sign in the pixels. A red rectangle would let us fake the demo with mocks and then fall
over the moment a real model looks at it.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

# Arrows are DRAWN, not typed. The unicode arrow glyph (U+2190) is missing from many system
# fonts and renders as an empty tofu box — which is exactly what happened on MSI. Geometry has
# no font dependency, and the detector has to see an arrow that actually looks like one.
ARROW_DIRS = ("left", "right", "up", "down")


def _draw_arrow(d, cx, cy, length, direction, fill):
    """A real arrow: shaft + solid head, drawn with polygons."""
    h = length / 2.0
    w = max(3, int(length * 0.13))          # shaft thickness
    head = length * 0.42                    # head length
    hw = length * 0.30                      # head half-width
    if direction == "left":
        d.line([(cx - h + head * 0.7, cy), (cx + h, cy)], fill=fill, width=w)
        d.polygon([(cx - h, cy), (cx - h + head, cy - hw), (cx - h + head, cy + hw)], fill=fill)
    elif direction == "right":
        d.line([(cx - h, cy), (cx + h - head * 0.7, cy)], fill=fill, width=w)
        d.polygon([(cx + h, cy), (cx + h - head, cy - hw), (cx + h - head, cy + hw)], fill=fill)
    elif direction == "up":
        d.line([(cx, cy + h), (cx, cy - h + head * 0.7)], fill=fill, width=w)
        d.polygon([(cx, cy - h), (cx - hw, cy - h + head), (cx + hw, cy - h + head)], fill=fill)
    elif direction == "down":
        d.line([(cx, cy - h), (cx, cy + h - head * 0.7)], fill=fill, width=w)
        d.polygon([(cx, cy + h), (cx - hw, cy + h - head), (cx + hw, cy + h - head)], fill=fill)


def _font(size: int):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/gnu-free/FreeSansBold.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)      # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def make_sign_texture(text: str, out_path: str, arrow: Optional[str] = None,
                      size: Tuple[int, int] = (1024, 512),
                      bg=(20, 90, 160), fg=(255, 255, 255)) -> str:
    """A blue directory sign with white text — the standard indoor wayfinding look."""
    from PIL import Image, ImageDraw

    W, H = size
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, W - 8, H - 8], outline=fg, width=6)

    label = text.upper()
    f_txt = _font(int(H * 0.22))
    arrow = arrow if arrow in ARROW_DIRS else None

    tw = d.textlength(label, font=f_txt)
    alen = H * 0.34 if arrow else 0
    gap = H * 0.10 if arrow else 0
    total = tw + gap + alen
    x = (W - total) / 2
    try:
        _, top, _, bot = d.textbbox((0, 0), label, font=f_txt)
        ty = (H - (bot - top)) / 2 - top
    except Exception:
        ty = H * 0.38
    d.text((x, ty), label, font=f_txt, fill=fg)
    if arrow:
        _draw_arrow(d, x + tw + gap + alen / 2, H / 2, alen, arrow, fg)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    return out_path


def add_sign_to_stage(stage, prim_path: str, texture_path: str,
                      position=(0.0, 0.0, 1.6), yaw_deg: float = 180.0,
                      width: float = 0.8, height: float = 0.4):
    """Hang a textured quad at a world pose.

    `yaw_deg` is the direction the sign FACES (its normal), in the same convention as the robot:
    0 = facing +X, 90 = facing +Y. A sign on the east wall that a robot driving +X should read
    faces back at it: yaw_deg=180.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    hw, hh = width / 2.0, height / 2.0
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    # quad in the local YZ plane, normal along +X (matches our "0 deg = faces +X")
    mesh.CreatePointsAttr([Gf.Vec3f(0, hw, -hh), Gf.Vec3f(0, -hw, -hh),
                           Gf.Vec3f(0, -hw, hh), Gf.Vec3f(0, hw, hh)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(1, 0, 0)] * 4)
    mesh.CreateExtentAttr([Gf.Vec3f(0, -hw, -hh), Gf.Vec3f(0, hw, hh)])
    # UVs. Viewed from +X (the direction the sign faces), the viewer's right is -Y. Vertex 0 is
    # at +Y = the viewer's LEFT, so it needs u=1 and the -Y vertices need u=0. Getting this
    # backwards renders the text mirrored, which is exactly what happened the first time.
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying)
    st.Set([Gf.Vec2f(1, 0), Gf.Vec2f(0, 0), Gf.Vec2f(0, 1), Gf.Vec2f(1, 1)])

    x = UsdGeom.Xformable(mesh)
    x.AddTranslateOp().Set(Gf.Vec3d(*position))
    x.AddRotateZOp().Set(float(yaw_deg))

    mat_path = prim_path + "/mat"
    mat = UsdShade.Material.Define(stage, mat_path)
    pbr = UsdShade.Shader.Define(stage, mat_path + "/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

    reader = UsdShade.Shader.Define(stage, mat_path + "/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex = UsdShade.Shader.Define(stage, mat_path + "/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.abspath(texture_path))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb")

    UsdShade.MaterialBindingAPI(mesh).Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh).Bind(mat)
    return mesh