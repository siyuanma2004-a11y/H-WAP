"""Versioned actions: metres, seconds, and scalar-first wxyz quaternions.

heading_at_start freezes the base XY origin and yaw when a chunk starts.
Its Z origin is the floor. No pose frame silently follows the moving robot.
"""
from dataclasses import asdict, dataclass
from typing import Optional
import numpy as np


def vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return tuple(float(x) for x in array)


def slerp(first, second, alpha):
    a, b = np.asarray(first, float), np.asarray(second, float)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + alpha * (b - a)
    else:
        theta = np.arccos(np.clip(dot, -1, 1))
        q = (np.sin((1-alpha)*theta)*a + np.sin(alpha*theta)*b)/np.sin(theta)
    return tuple(q / np.linalg.norm(q))


@dataclass(frozen=True)
class Pose:
    position: tuple
    quaternion: Optional[tuple] = None

    def __post_init__(self):
        object.__setattr__(self, "position", vector(self.position, 3, "position"))
        if self.quaternion is not None:
            q = np.asarray(vector(self.quaternion, 4, "quaternion"))
            if not 0.99 <= np.linalg.norm(q) <= 1.01:
                raise ValueError("quaternion must have unit norm (wxyz)")
            object.__setattr__(self, "quaternion", tuple(q / np.linalg.norm(q)))

    @classmethod
    def parse(cls, value):
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) - {"position", "quaternion"}:
            raise ValueError("invalid pose fields")
        return cls(**value)


@dataclass(frozen=True)
class Keyframe:
    t: float
    left_wrist: Optional[Pose] = None
    right_wrist: Optional[Pose] = None
    left_hand: Optional[float] = None
    right_hand: Optional[float] = None
    velocity: tuple = (0.0, 0.0, 0.0)
    head: Optional[Pose] = None
    waist_height: Optional[float] = None

    def __post_init__(self):
        if not np.isfinite(self.t) or self.t < 0:
            raise ValueError("keyframe time must be non-negative and finite")
        v = vector(self.velocity, 3, "velocity")
        if np.any(np.abs(v) > np.array([0.15, 0.10, 0.30]) + 1e-12):
            raise ValueError("velocity exceeds configured simulation limits")
        object.__setattr__(self, "velocity", v)
        for side in ("left", "right"):
            grip = getattr(self, side + "_hand")
            if grip is not None and (not np.isfinite(grip) or not 0 <= grip <= 1):
                raise ValueError("hand closure must be between 0 and 1")
        if self.waist_height is not None and (not np.isfinite(self.waist_height) or not 0.3 <= self.waist_height <= 0.85):
            raise ValueError("waist height must be in [0.3, 0.85] metres")

    @classmethod
    def parse(cls, value):
        item = dict(value)
        for key in ("left_wrist", "right_wrist", "head"):
            if key in item:
                item[key] = Pose.parse(item[key])
        try:
            return cls(**item)
        except TypeError as error:
            raise ValueError(f"invalid keyframe: {error}") from error


@dataclass(frozen=True)
class CommandChunk:
    name: str
    keyframes: tuple
    frame: str = "heading_at_start"
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.frame not in {"heading_at_start", "world"}:
            raise ValueError("unsupported action schema or frame")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("chunk name is required")
        if not 2 <= len(self.keyframes) <= 64:
            raise ValueError("a chunk requires 2 to 64 keyframes")
        times = [k.t for k in self.keyframes]
        if times[0] != 0 or np.any(np.diff(times) <= 0) or not 0 < times[-1] <= 3:
            raise ValueError("keyframes must start at zero, increase, and end within 3 seconds")
        for key in ("left_wrist", "right_wrist", "head", "left_hand", "right_hand", "waist_height"):
            if getattr(self.keyframes[0], key) is None and any(getattr(k, key) is not None for k in self.keyframes[1:]):
                raise ValueError(f"{key} must be initialized at time zero")

    @property
    def duration(self):
        return float(self.keyframes[-1].t)

    @classmethod
    def parse(cls, value):
        item = dict(value)
        item["keyframes"] = tuple(Keyframe.parse(k) for k in item.get("keyframes", []))
        try:
            return cls(**item)
        except TypeError as error:
            raise ValueError(f"invalid chunk: {error}") from error

    def to_dict(self):
        return asdict(self)

    def sample(self, t):
        t = float(np.clip(t, 0, self.duration))
        index = min(int(np.searchsorted([k.t for k in self.keyframes], t, side="right"))-1, len(self.keyframes)-2)
        resolved, previous = [], {}
        for k in self.keyframes[:index+2]:
            fields = {key: getattr(k, key) for key in ("left_wrist", "right_wrist", "head", "left_hand", "right_hand", "waist_height")}
            for key, value in fields.items():
                if value is not None:
                    previous[key] = value
                fields[key] = previous.get(key)
            resolved.append(fields)
        first, second = self.keyframes[index:index+2]
        a, b = resolved[index:index+2]
        alpha = (t-first.t)/(second.t-first.t)
        fields = {}
        for key in ("left_wrist", "right_wrist", "head"):
            if a[key] is None:
                fields[key] = None
            else:
                q = None if a[key].quaternion is None or b[key].quaternion is None else slerp(a[key].quaternion, b[key].quaternion, alpha)
                fields[key] = Pose(tuple((1-alpha)*np.array(a[key].position)+alpha*np.array(b[key].position)), q)
        for key in ("left_hand", "right_hand", "waist_height"):
            fields[key] = None if a[key] is None else (1-alpha)*a[key]+alpha*b[key]
        fields["velocity"] = tuple((1-alpha)*np.array(first.velocity)+alpha*np.array(second.velocity))
        return Keyframe(t=t, **fields)
