"""Lossless RGB PNG output without an additional image dependency."""
import struct
import zlib
import numpy as np


def png_bytes(rgb):
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected RGB image")
    h, w = rgb.shape[:2]
    def block(kind, data):
        return struct.pack("!I", len(data))+kind+data+struct.pack("!I", zlib.crc32(kind+data)&0xffffffff)
    rows = b"".join(b"\x00"+row.tobytes() for row in rgb)
    return b"\x89PNG\r\n\x1a\n"+block(b"IHDR", struct.pack("!2I5B", w,h,8,2,0,0,0))+block(b"IDAT",zlib.compress(rows))+block(b"IEND",b"")
