"""Generate the anti-aliased glyph atlas used by wotan/plot (font.odin).

Renders DejaVu Sans (regular + bold) at a few pixel sizes with FreeType hinting
via Pillow and writes a compact little-endian blob, embedded with #load:

    "WFNT" u16 n_fonts
    per font:  u8 size_px, u8 bold, i16 ascent, i16 descent, u16 n_glyphs,
               u32 data_size, then n_glyphs records:
                   u32 codepoint, u16 advance (1/64 px), i16 x_off, i16 y_off,
                   u8 w, u8 h, u32 offset   (offset into this font's pixel data)
               then data_size bytes of 8-bit coverage, rows top to bottom
    x_off / y_off place the bitmap relative to the pen position at the top of
    the line box (ascender line).

Usage: python gen_font.py [regular.ttf bold.ttf]   (defaults: matplotlib's DejaVu)
DejaVu fonts: Bitstream Vera license + public domain changes, see LICENSE_DEJAVU.
"""

import os
import struct
import sys

from PIL import ImageFont

REGULAR_SIZES = [10, 11, 13, 16, 20, 24]
BOLD_SIZES = [13, 16, 20, 24]
EXTRA = "αβγδεθλμνπρστφωΓΔΣΦΩ²³°±×÷·−–—≈≠≤≥∞√∑€£…‰"
CHARS = [chr(c) for c in range(32, 127)] + list(EXTRA)


def default_paths():
    import matplotlib

    base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    return os.path.join(base, "DejaVuSans.ttf"), os.path.join(base, "DejaVuSans-Bold.ttf")


def build_font(path, size, bold):
    font = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.BASIC)
    ascent, descent = font.getmetrics()
    records, data = [], bytearray()
    for ch in CHARS:
        advance = int(round(font.getlength(ch) * 64))
        mask, (ox, oy) = font.getmask2(ch, mode="L", anchor="la")
        w, h = mask.size
        if w == 0 or h == 0 or ch == " ":
            records.append((ord(ch), advance, 0, 0, 0, 0, len(data)))
            continue
        assert w < 256 and h < 256
        pixels = bytes(mask)
        records.append((ord(ch), advance, ox, oy, w, h, len(data)))
        data += pixels
    out = bytearray(struct.pack("<BBhhHI", size, bold, ascent, descent, len(records), len(data)))
    for r in records:
        out += struct.pack("<IHhhBBI", *r)
    out += data
    return out


def main():
    regular, bold = sys.argv[1:3] if len(sys.argv) >= 3 else default_paths()
    fonts = [build_font(regular, s, 0) for s in REGULAR_SIZES] + [build_font(bold, s, 1) for s in BOLD_SIZES]
    blob = bytearray(b"WFNT") + struct.pack("<H", len(fonts))
    for f in fonts:
        blob += f
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dejavu_sans.bin")
    with open(out, "wb") as fh:
        fh.write(blob)
    print(f"wrote {out}: {len(blob)} bytes, {len(fonts)} fonts, {len(CHARS)} glyphs each")


if __name__ == "__main__":
    main()
