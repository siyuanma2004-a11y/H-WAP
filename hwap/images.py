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


def decode_png(data):
    """Decode 8-bit non-interlaced greyscale/RGB/RGBA PNG to RGB uint8."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    offset, idat, header = 8, b"", None
    while offset < len(data):
        length, kind = struct.unpack("!I4s", data[offset:offset+8])
        body = data[offset+8:offset+8+length]
        offset += 12+length
        if kind == b"IHDR":
            header = struct.unpack("!2I5B", body)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
    w, h, depth, color, _, _, interlace = header
    channels = {0: 1, 2: 3, 6: 4}.get(color)
    if depth != 8 or channels is None or interlace:
        raise ValueError("unsupported PNG format")
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1+w*channels)
    out = np.zeros((h, w*channels), np.int32)
    previous = np.zeros(w*channels, np.int32)
    for y in range(h):
        kind, row = raw[y, 0], raw[y, 1:].astype(np.int32)
        if kind == 0:
            line = row
        elif kind == 1:
            line = (np.cumsum(row.reshape(w, channels), axis=0) & 0xff).reshape(-1)
        elif kind == 2:
            line = (row+previous) & 0xff
        else:
            line = np.zeros_like(row)
        for x in range(len(row) if kind > 2 else 0):
            a = line[x-channels] if x >= channels else 0
            b = previous[x]
            c = previous[x-channels] if x >= channels else 0
            if kind == 0: predictor = 0
            elif kind == 1: predictor = a
            elif kind == 2: predictor = b
            elif kind == 3: predictor = (a+b)//2
            else:
                p = a+b-c
                pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)
                predictor = a if pa <= pb and pa <= pc else b if pb <= pc else c
            line[x] = (row[x]+predictor) & 0xff
        out[y], previous = line, line
    image = out.reshape(h, w, channels).astype(np.uint8)
    return np.repeat(image, 3, axis=2) if channels == 1 else image[..., :3]
