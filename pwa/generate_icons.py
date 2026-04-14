# -*- coding: utf-8 -*-
"""
generate_icons.py
─────────────────
Generates PWA icons (192x192 and 512x512) for the AI Malaria Lab Assistant
using only Pillow — no internet required.
Run once: python generate_icons.py
"""

from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Brand colours (matching the app)
BG_GRAD_TOP    = (0, 168, 150)   # #00A896
BG_GRAD_BOTTOM = (2, 128, 144)   # #028090
WHITE          = (255, 255, 255)


def make_icon(size: int, filename: str):
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── Rounded rectangle background (gradient-like via two rects) ──
    r = size // 5  # corner radius
    # Bottom colour fill
    draw.rounded_rectangle([0, 0, size, size], radius=r, fill=BG_GRAD_BOTTOM)
    # Top-half lighter tone
    draw.rounded_rectangle([0, 0, size, size // 2 + r], radius=r, fill=BG_GRAD_TOP)
    # Re-fill bottom half blending overlap
    draw.rounded_rectangle([0, size // 2 - r, size, size], radius=r, fill=BG_GRAD_BOTTOM)

    # ── Stethoscope / medical cross symbol ──────────────────────────
    cx, cy = size // 2, size // 2
    cross_arm = size // 5
    thick     = max(size // 10, 4)

    # Vertical arm
    draw.rounded_rectangle(
        [cx - thick // 2, cy - cross_arm, cx + thick // 2, cy + cross_arm],
        radius=thick // 2,
        fill=WHITE,
    )
    # Horizontal arm
    draw.rounded_rectangle(
        [cx - cross_arm, cy - thick // 2, cx + cross_arm, cy + thick // 2],
        radius=thick // 2,
        fill=WHITE,
    )

    # Small circle in the very centre (DNA / lens feel)
    dot_r = max(size // 16, 3)
    draw.ellipse(
        [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
        fill=(0, 168, 150),
    )

    # Save
    out_path = os.path.join(OUTPUT_DIR, filename)
    img.save(out_path, "PNG")
    print(f"  [OK] Saved {filename}  ({size}x{size})")


if __name__ == "__main__":
    print("Generating PWA icons...")
    make_icon(192, "icon-192.png")
    make_icon(512, "icon-512.png")
    print("Done - icons saved to:", OUTPUT_DIR)
