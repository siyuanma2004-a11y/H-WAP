"""Per-frame action vectors and pose images for action-conditioned video models.

Vectors use one fixed world-frame layout (ACTION_LAYOUT) for both the
commanded stream (recorded chunks sampled at frame times) and the achieved
stream (forward kinematics of the recorded qpos). A boolean mask marks fields a
chunk left unset. Pose images come in two forms for the proposal ablation:
``achieved_skeleton_image`` (measured / SONIC-in-sim bodies) and
``command_pose_image`` (commanded targets plus a commanded base path only).
"""
import numpy as np

ACTION_LAYOUT = (
    ("left_palm_position", 3), ("left_palm_quaternion", 4),
    ("right_palm_position", 3), ("right_palm_quaternion", 4),
    ("left_hand", 1), ("right_hand", 1),
    ("head_position", 3), ("head_quaternion", 4),
    ("base_velocity", 3), ("waist_height", 1),
)
SLICES, _start = {}, 0
for _name, _size in ACTION_LAYOUT:
    SLICES[_name] = slice(_start, _start+_size)
    _start += _size
ACTION_DIM = _start
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
HEAD_OFFSET = np.array([0, 0, 0.35])  # torso_link reference used by the SONIC adapter
COLORS = {"skeleton": (255, 180, 50), "left": (80, 160, 255), "right": (255, 90, 90),
          "head": (90, 255, 120), "base": (230, 230, 230)}


def describe_layout():
    return [{"name": n, "start": SLICES[n].start, "stop": SLICES[n].stop} for n, _ in ACTION_LAYOUT]


def yaw_quaternion(yaw):
    return np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])


def quaternion_multiply(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def quaternion_matrix(q):
    w, x, y, z = np.asarray(q, float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def world_pose(position, quaternion, anchor, frame):
    """Same convention as HumanoidEnvironment._world_pose (no env import)."""
    position = None if position is None else np.asarray(position, float)
    quaternion = None if quaternion is None else np.asarray(quaternion, float)
    if frame == "world":
        return position, quaternion
    x, y, yaw = anchor
    if position is not None:
        c, s = np.cos(yaw), np.sin(yaw)
        position = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ position + [x, y, 0]
    if quaternion is not None:
        quaternion = quaternion_multiply(yaw_quaternion(yaw), quaternion)
    return position, quaternion


def active_chunk(chunks, row):
    """The chunk whose command is in effect at a recorded row.

    A row shared by two chunks (end of one, start of the next) belongs to the
    finished chunk; rows before the first chunk use that chunk at t=0.
    """
    for item in chunks:
        if item["start_row"] < row <= item["end_row"]:
            return item
    for item in chunks:
        if item["start_row"] == row:
            return item
    return None


def command_vector(chunk, t, anchor):
    """Return (vector, mask) for a parsed CommandChunk sampled at chunk time t."""
    key = chunk.sample(t)
    vector, mask = np.zeros(ACTION_DIM), np.zeros(ACTION_DIM, dtype=bool)
    def put(name, value):
        if value is not None:
            vector[SLICES[name]], mask[SLICES[name]] = value, True
    for side in ("left", "right"):
        pose = getattr(key, side+"_wrist")
        if pose is not None:
            p, q = world_pose(pose.position, pose.quaternion, anchor, chunk.frame)
            put(side+"_palm_position", p)
            put(side+"_palm_quaternion", q)
        put(side+"_hand", getattr(key, side+"_hand"))
    head = getattr(key, "head", None)
    if head is not None:
        p, q = world_pose(getattr(head, "position", None), getattr(head, "quaternion", None), anchor, chunk.frame)
        put("head_position", p)
        put("head_quaternion", q)
    put("base_velocity", np.asarray(key.velocity, float))
    put("waist_height", getattr(key, "waist_height", None))
    return vector, mask


def commanded_vectors(record, arrays, rows):
    """Commanded vectors at recorded rows; heading_at_start uses the recorded start base pose."""
    from .actions import CommandChunk
    parsed = [CommandChunk.parse(item["chunk"]) for item in record["chunks"]]
    vectors, masks = np.zeros((len(rows), ACTION_DIM)), np.zeros((len(rows), ACTION_DIM), dtype=bool)
    for i, row in enumerate(rows):
        item = active_chunk(record["chunks"], row)
        if item is None:
            continue
        chunk = parsed[record["chunks"].index(item)]
        t = float(arrays["time"][row]-arrays["time"][item["start_row"]])
        vectors[i], masks[i] = command_vector(chunk, t, arrays["base_pose"][item["start_row"]])
    return vectors, masks


class KinematicsModel:
    """Forward kinematics of recorded qpos with the current G1 scene."""

    def __init__(self):
        import mujoco
        from agent_g1.simulation import G1Simulation
        self.mujoco, self.sim = mujoco, G1Simulation()
        m = self.sim.model
        name = lambda kind, n: self.sim._required_id(kind, n)
        self.torso = name(mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.shoulders = {s: name(mujoco.mjtObj.mjOBJ_BODY, f"{s}_shoulder_pitch_link") for s in ("left", "right")}
        self.palms = {"left": self.sim.left_palm_site_id, "right": self.sim.right_palm_site_id}
        self.hands = {}
        for side in ("left", "right"):
            joints = [name(mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_{s}_joint") for s in HAND_JOINTS]
            lo, hi = m.jnt_range[joints, 0], m.jnt_range[joints, 1]
            opened = np.clip(0, lo, hi)
            closed = np.where(np.abs(lo) > np.abs(hi), lo, hi)
            self.hands[side] = (m.jnt_qposadr[joints], opened, 0.65*(closed-opened))

    def forward(self, qpos):
        m, d = self.sim.model, self.sim.data
        d.qpos[:] = qpos
        self.mujoco.mj_kinematics(m, d)

    def achieved(self, qpos, qvel):
        """Achieved vector from one recorded state (all fields valid)."""
        self.forward(qpos)
        d, vector = self.sim.data, np.zeros(ACTION_DIM)
        for side, site in self.palms.items():
            q = np.empty(4)
            self.mujoco.mju_mat2Quat(q, d.site_xmat[site])
            vector[SLICES[side+"_palm_position"]] = d.site_xpos[site]
            vector[SLICES[side+"_palm_quaternion"]] = q
            addresses, opened, span = self.hands[side]
            vector[SLICES[side+"_hand"]] = np.clip(np.mean((qpos[addresses]-opened)/span), 0, 1)
        R = d.xmat[self.torso].reshape(3, 3)
        vector[SLICES["head_position"]] = d.xpos[self.torso]+R@HEAD_OFFSET
        vector[SLICES["head_quaternion"]] = d.xquat[self.torso]
        base_R = quaternion_matrix(qpos[3:7])
        yaw = np.arctan2(base_R[1, 0], base_R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        vx, vy = c*qvel[0]+s*qvel[1], -s*qvel[0]+c*qvel[1]
        vector[SLICES["base_velocity"]] = [vx, vy, (base_R@qvel[3:6])[2]]
        vector[SLICES["waist_height"]] = d.xpos[self.sim.pelvis_body_id, 2]
        return vector

    def shoulders_relative(self, qpos):
        """Shoulder positions in the base heading frame (for command pose images)."""
        self.forward(qpos)
        R = quaternion_matrix(qpos[3:7])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        inverse = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        base = np.array([qpos[0], qpos[1], 0])
        return {side: inverse@(self.sim.data.xpos[b]-base) for side, b in self.shoulders.items()}


def achieved_vectors(arrays, rows, kinematics=None):
    kinematics = kinematics or KinematicsModel()
    vectors = np.stack([kinematics.achieved(arrays["qpos"][r], arrays["qvel"][r]) for r in rows])
    return vectors, np.ones_like(vectors, dtype=bool)


def integrate_base_path(start_pose, times, velocities, dt=0.02):
    """Commanded base path: integrate heading-frame [vx, vy, yaw_rate] from a start pose.

    ``velocities[i]`` holds over [times[i], times[i+1]); returns (len(times), 3) [x, y, yaw].
    """
    x, y, yaw = map(float, start_pose)
    path = [(x, y, yaw)]
    for i in range(1, len(times)):
        span = float(times[i]-times[i-1])
        steps = max(1, int(np.ceil(span/dt-1e-9)))
        vx, vy, w = velocities[i-1]
        for _ in range(steps):
            h = span/steps
            x += h*(np.cos(yaw)*vx-np.sin(yaw)*vy)
            y += h*(np.sin(yaw)*vx+np.cos(yaw)*vy)
            yaw += h*w
        path.append((x, y, yaw))
    return np.array(path)


def project(points, camera):
    """Pinhole projection used by every pose image: returns (pixels (N,2), depth (N,))."""
    points = np.atleast_2d(np.asarray(points, float))
    K = np.asarray(camera["intrinsics"], float)
    local = (points-np.array(camera["position"])) @ np.array(camera["rotation_world_from_camera"])
    depth = -local[:, 2]
    safe = np.maximum(depth, 1e-6)
    return np.column_stack([K[0, 2]+K[0, 0]*local[:, 0]/safe, K[1, 2]-K[1, 1]*local[:, 1]/safe]), depth


def _segment(image, a, b, color):
    h, w = image.shape[:2]
    if np.any(np.abs([a, b]) > max(w, h)*4):
        return
    count = min(4*max(w, h), max(2, int(np.linalg.norm(a-b))+1))
    points = np.rint(np.linspace(a, b, count)).astype(int)
    valid = (points[:, 0] >= 0)&(points[:, 0] < w)&(points[:, 1] >= 0)&(points[:, 1] < h)
    image[points[valid, 1], points[valid, 0]] = color


def _dot(image, p, color, radius):
    h, w = image.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    image[(xx-p[0])**2+(yy-p[1])**2 <= radius**2] = color


def _polyline(image, points, camera, color):
    pixels, depth = project(points, camera)
    for i in range(len(points)-1):
        if depth[i] > 0.02 and depth[i+1] > 0.02:
            _segment(image, pixels[i], pixels[i+1], color)


def achieved_skeleton_image(positions, ids, parents, camera):
    """Measured robot bodies drawn as parent-child segments (robot only)."""
    image = np.zeros((camera["height"], camera["width"], 3), dtype=np.uint8)
    pixels, depth = project(positions, camera)
    lookup = {body: i for i, body in enumerate(ids)}
    for i, parent in enumerate(parents):
        j = lookup.get(parent)
        if j is None or depth[i] <= 0.02 or depth[j] <= 0.02:
            continue
        _segment(image, pixels[i], pixels[j], COLORS["skeleton"])
    return image


def command_pose_image(points, camera):
    """Draw commanded targets only.

    ``points`` keys (each optional, world metres): left_palm/right_palm targets,
    left_shoulder/right_shoulder estimates, head and head_forward (ray end), and
    base_path (N,3) on the floor.
    """
    image = np.zeros((camera["height"], camera["width"], 3), dtype=np.uint8)
    radius = max(1, camera["width"]//64)
    get = lambda key: None if points.get(key) is None else np.asarray(points[key], float)
    if get("base_path") is not None and len(points["base_path"]) > 1:
        _polyline(image, get("base_path"), camera, COLORS["base"])
    if get("head") is not None and get("head_forward") is not None:
        _polyline(image, np.stack([get("head"), get("head_forward")]), camera, COLORS["head"])
    for side in ("left", "right"):
        palm, shoulder = get(side+"_palm"), get(side+"_shoulder")
        if palm is None:
            continue
        if shoulder is not None:
            _polyline(image, np.stack([shoulder, palm]), camera, COLORS[side])
        pixel, depth = project(palm, camera)
        if depth[0] > 0.02:
            _dot(image, pixel[0], COLORS[side], radius)
    return image


def command_pose_points(vector, mask, base_pose, shoulders_relative, base_path=None):
    """World points for command_pose_image from one commanded vector.

    ``base_pose`` is the commanded (integrated) [x, y, yaw]; shoulders come from
    the clip's first (history) frame expressed in the heading frame, so no future
    measurement enters the condition.
    """
    x, y, yaw = base_pose
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    points = {"base_path": None if base_path is None else np.column_stack([base_path[:, :2], np.full(len(base_path), 0.02)])}
    for side in ("left", "right"):
        valid = mask[SLICES[side+"_palm_position"]].all()
        points[side+"_palm"] = vector[SLICES[side+"_palm_position"]] if valid else None
        points[side+"_shoulder"] = R@shoulders_relative[side]+[x, y, 0] if shoulders_relative else None
    if mask[SLICES["head_quaternion"]].all():
        head = vector[SLICES["head_position"]] if mask[SLICES["head_position"]].all() else \
            (R@(0.5*(shoulders_relative["left"]+shoulders_relative["right"]))+[x, y, 0.1] if shoulders_relative else None)
        if head is not None:
            points["head"] = head
            points["head_forward"] = head+0.25*quaternion_matrix(vector[SLICES["head_quaternion"]])[:, 0]
    return points
