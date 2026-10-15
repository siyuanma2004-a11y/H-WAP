---
name: reach-visible-marker
description: Touch a visible marker with the right palm while standing still, from camera images and calibration.
uses: 0
successes: 0
---
When to use
- The target is visible in at least one camera and within reach while standing (no walking needed).

How to do it
1. Estimate the marker's 3D world position from the images and the public camera calibration
   (camera axes: X right, Y up, looking along -Z; pixels start top-left). Match the marker across
   views when more than one view is given.
2. Write one world-frame chunk with three right_wrist keyframes: t=0 at the measured
   robot.right_wrist, t=1.5 at the estimate, t=2.5 holding the estimate. Leave the other fields unset
   and keep velocity zero. `scripts/straight_reach.py` makes these keyframes.
3. After execution, compare the measured palm with the marker in the new images and send a short
   corrective chunk if they do not overlap.

Useful numbers (this MuJoCo G1 scene)
- right_wrist means the palm site, 10 cm beyond the wrist yaw joint, not the wrist joint.
- Wrist targets must stay within 0.9 m (XY) of the base and between 0.55 and 1.55 m high.
- The standing right palm starts near [0.00, -0.23, 0.62] m with the arm hanging.
- The tabletop surface is at about z = 0.75 m; the reach marker floats above it.
- Success needs the palm within 4.5 cm of the marker for 0.2 s; a 1.5 s reach plus a 1 s hold works.

Examples
- `examples/reach_world_frame.json`: a successful reach; its target was the model's visual
  estimate, not simulator truth.

Common failures
- Treating the blue cube as the target.
- Aiming at the wrist joint instead of the palm (about 10 cm short).
- Paths that sweep through the table: approach from above the tabletop height.
