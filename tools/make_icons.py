"""Generate the RVC Android launcher icons + README logo.

Motif: audio waveform + conversion (swap) arrows on a violet -> indigo gradient.
Renders a high-res master, then downsamples to every mipmap density plus the
adaptive icon layers (background / foreground / monochrome) and a docs/ logo.
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = r"C:\Users\hurwy\Codes\rvc-android"
RES = os.path.join(ROOT, "app", "src", "main", "res")
DOCS = os.path.join(ROOT, "docs", "img")
os.makedirs(DOCS, exist_ok=True)

M = 1024  # master canvas
SS = 4    # supersample factor for crisp shapes

# --- palette -----------------------------------------------------------------
VIOLET = (124, 77, 255)     # #7C4DFF
INDIGO = (43, 22, 120)      # #2B1678
ACCENT = (199, 186, 255)    # light violet for the arrows
WHITE = (255, 255, 255)


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(len(a)))


def gradient_bg(size):
    """Diagonal violet (top-left) -> indigo (bottom-right)."""
    base = Image.new("RGB", (size, size))
    px = base.load()
    last = size - 1
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * last)
            px[x, y] = lerp(VIOLET, INDIGO, t)
    return base.convert("RGBA")


def draw_arrow(d, x_start, x_end, y, th, head_l, head_w, color):
    direction = 1 if x_end >= x_start else -1
    base_x = x_end - direction * head_l          # triangle base (where head meets shaft)
    shaft_to = base_x + direction * th           # run the shaft past the base; the head hides it
    x0, x1 = sorted((x_start, shaft_to))
    d.rounded_rectangle([x0, y - th / 2, x1, y + th / 2], radius=th / 2, fill=color)
    d.polygon([(x_end, y), (base_x, y - head_w), (base_x, y + head_w)], fill=color)


def foreground_art(size, mono=False):
    """Waveform bars (white) + swap arrows (accent), centred, on transparent."""
    s = size * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    arrow_col = WHITE if mono else ACCENT
    cx = s / 2

    # waveform — 7 rounded bars, tallest in the middle
    heights = [0.18, 0.34, 0.50, 0.38, 0.50, 0.30, 0.20]
    n = len(heights)
    bw = 0.052 * s
    gap = 0.026 * s
    total = n * bw + (n - 1) * gap
    x0 = cx - total / 2
    cyw = 0.38 * s
    for i, h in enumerate(heights):
        bh = h * s
        left = x0 + i * (bw + gap)
        top = cyw - bh / 2
        d.rounded_rectangle([left, top, left + bw, top + bh], radius=bw / 2, fill=WHITE)

    # conversion swap arrows (top -> right, bottom <- left)
    th = 0.040 * s
    head_l = 0.075 * s
    head_w = 0.060 * s
    span = 0.20 * s
    cya = 0.74 * s
    dy = 0.058 * s
    draw_arrow(d, cx - span, cx + span, cya - dy, th, head_l, head_w, arrow_col)
    draw_arrow(d, cx + span, cx - span, cya + dy, th, head_l, head_w, arrow_col)

    return img.resize((size, size), Image.LANCZOS)


def rounded_mask(size, radius):
    m = Image.new("L", (size * SS, size * SS), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        [0, 0, size * SS - 1, size * SS - 1], radius=radius * SS, fill=255
    )
    return m.resize((size, size), Image.LANCZOS)


def circle_mask(size):
    m = Image.new("L", (size * SS, size * SS), 0)
    ImageDraw.Draw(m).ellipse([0, 0, size * SS - 1, size * SS - 1], fill=255)
    return m.resize((size, size), Image.LANCZOS)


# --- masters -----------------------------------------------------------------
BG = gradient_bg(M)
FG = foreground_art(M)          # legacy: art fills more of the tile
FG_MONO = foreground_art(M, mono=True)

# composited full-tile icon (background + foreground), used for legacy + docs
ICON = BG.copy()
ICON.alpha_composite(FG)

ADAPTIVE = {"mdpi": 108, "hdpi": 162, "xhdpi": 216, "xxhdpi": 324, "xxxhdpi": 432}
LEGACY = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}


def save_png(img, path):
    img.save(path, "PNG")


def save_webp(img, path):
    img.save(path, "WEBP", lossless=True, quality=100)


for dens, sz in ADAPTIVE.items():
    folder = os.path.join(RES, f"mipmap-{dens}")
    os.makedirs(folder, exist_ok=True)
    save_png(BG.resize((sz, sz), Image.LANCZOS), os.path.join(folder, "ic_launcher_background.png"))
    # foreground art lives in the central safe zone (scale to 80% of the tile)
    fg_tile = Image.new("RGBA", (M, M), (0, 0, 0, 0))
    inner = int(M * 0.80)
    off = (M - inner) // 2
    fg_tile.alpha_composite(FG.resize((inner, inner), Image.LANCZOS), (off, off))
    save_png(fg_tile.resize((sz, sz), Image.LANCZOS), os.path.join(folder, "ic_launcher_foreground.png"))
    mono_tile = Image.new("RGBA", (M, M), (0, 0, 0, 0))
    mono_tile.alpha_composite(FG_MONO.resize((inner, inner), Image.LANCZOS), (off, off))
    save_png(mono_tile.resize((sz, sz), Image.LANCZOS), os.path.join(folder, "ic_launcher_monochrome.png"))

for dens, sz in LEGACY.items():
    folder = os.path.join(RES, f"mipmap-{dens}")
    sq = ICON.resize((sz, sz), Image.LANCZOS)
    sq.putalpha(rounded_mask(sz, sz * 0.22))
    save_webp(sq, os.path.join(folder, "ic_launcher.webp"))
    rd = ICON.resize((sz, sz), Image.LANCZOS)
    rd.putalpha(circle_mask(sz))
    save_webp(rd, os.path.join(folder, "ic_launcher_round.webp"))

# --- docs assets -------------------------------------------------------------
icon512 = ICON.resize((512, 512), Image.LANCZOS)
icon512.putalpha(rounded_mask(512, 512 * 0.22))
save_png(icon512, os.path.join(DOCS, "icon.png"))


def load_font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


# README logo banner: icon + wordmark, transparent background
LW, LH = 1040, 300
logo = Image.new("RGBA", (LW, LH), (0, 0, 0, 0))
icon_sz = 236
icon_logo = ICON.resize((icon_sz, icon_sz), Image.LANCZOS)
icon_logo.putalpha(rounded_mask(icon_sz, icon_sz * 0.24))
iy = (LH - icon_sz) // 2
logo.alpha_composite(icon_logo, (40, iy))

d = ImageDraw.Draw(logo)
title_font = load_font(["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"], 116)
sub_font = load_font(["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"], 44)
tx = 40 + icon_sz + 44
title = "RVC Android"
sub = "On-device voice conversion"
tb = d.textbbox((0, 0), title, font=title_font)
sb = d.textbbox((0, 0), sub, font=sub_font)
th_t = tb[3] - tb[1]
th_s = sb[3] - sb[1]
gap = 26
total = th_t + gap + th_s
top = (LH - total) / 2
# anchor="lm" -> (x, vertical-center); avoids font top-bearing math
d.text((tx, top + th_t / 2), title, font=title_font, fill=(108, 92, 231, 255), anchor="lm")
d.text((tx + 3, top + th_t + gap + th_s / 2), sub, font=sub_font, fill=(150, 150, 165, 255), anchor="lm")
save_png(logo, os.path.join(DOCS, "logo.png"))

print("done: launcher icons, docs/img/icon.png, docs/img/logo.png")
