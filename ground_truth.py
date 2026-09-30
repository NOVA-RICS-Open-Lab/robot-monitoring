"""
ground_truth.py — authoritative expected result for every clip in the study.

Provided by the researcher after a frame-by-frame review. Every clip is an
anomaly; this records *what kind* so a verdict can be scored as "right for the
right reason", and roughly *when* (as a fraction of the clip, robust to the
different frame counts) so two-stage localisation can be scored.

Fields per clip:
  track       : "external" | "internal"
  desc        : one-line description of the true anomaly
  human       : a human body part is (part of) the anomaly
  objects     : list of foreign physical objects that are (part of) the anomaly
  coco_class  : nearest MS-COCO class name for each object, "" if out-of-vocabulary
  mixed       : the clip is both an external intrusion and an internal fault
  internal_cause : for internal / mixed clips, the control-level fault
  window      : (start_frac, end_frac) of the clip where the anomaly is visible
                (for internal-only faults the "window" is the control event near
                 the end of the clip)
"""

GROUND_TRUTH = {
    # ---------------- EXTERNAL ----------------
    ("external", "an1"): dict(
        track="external", desc="Human hand leaves a white object, then withdraws",
        human=True, objects=["white object"], coco_class=[""], mixed=False,
        internal_cause=None, window=(0.70, 0.90)),
    ("external", "an2"): dict(
        track="external", desc="Human hand enters the workspace and leaves immediately",
        human=True, objects=[], coco_class=[], mixed=False,
        internal_cause=None, window=(0.25, 0.45)),
    ("external", "an3"): dict(
        track="external", desc="Human hand enters the workspace and touches the robot",
        human=True, objects=[], coco_class=[], mixed=False,
        internal_cause=None, window=(0.10, 0.95)),
    ("external", "an4"): dict(
        track="external", desc="Human hand enters and rearranges the yellow pieces",
        human=True, objects=[], coco_class=[], mixed=False,
        internal_cause=None, window=(0.15, 0.70)),
    ("external", "an5"): dict(
        track="external", desc="Tennis ball thrown into the environment",
        human=False, objects=["tennis ball"], coco_class=["sports ball"], mixed=False,
        internal_cause=None, window=(0.10, 0.35)),
    ("external", "an6"): dict(
        track="external", desc="A plant is placed in the environment",
        human=False, objects=["plant"], coco_class=["potted plant"], mixed=False,
        internal_cause=None, window=(0.20, 0.95)),
    ("external", "an7"): dict(
        track="external", desc="Human hand places a white object in the environment",
        human=True, objects=["white object"], coco_class=[""], mixed=False,
        internal_cause=None, window=(0.15, 0.95)),
    ("external", "an8"): dict(
        track="external", desc="Human hand throws a black wallet; a white object is also present",
        human=True, objects=["black wallet", "white object"], coco_class=["handbag", ""], mixed=False,
        internal_cause=None, window=(0.10, 0.95)),
    ("external", "an9"): dict(
        track="external", desc="Plastic bottle in the environment",
        human=False, objects=["plastic bottle"], coco_class=["bottle"], mixed=False,
        internal_cause=None, window=(0.00, 1.00)),
    ("external", "an10"): dict(
        track="external", desc="Human hand rips a piece off the robot (mixed)",
        human=True, objects=[], coco_class=[], mixed=True,
        internal_cause="human removes piece", window=(0.40, 0.70)),

    # ---------------- INTERNAL (an1-an5 have a state log; an6-an10 are the SAME
    # five clips byte-for-byte but WITHOUT the log — a vision-only stress test) ----
    ("internal", "an1"): dict(
        track="internal", has_log=True, desc="Robot picks a piece that does not exist",
        human=False, objects=[], coco_class=[], mixed=False,
        internal_cause="phantom pick (gripper error, unexpected idle)", window=(0.60, 1.00)),
    ("internal", "an2"): dict(
        track="internal", has_log=True, desc="Human hand rips a piece off the robot (mixed)",
        human=True, objects=[], coco_class=[], mixed=True,
        internal_cause="human removes piece (gripper error, unexpected idle)", window=(0.25, 1.00)),
    ("internal", "an3"): dict(
        track="internal", has_log=True, desc="Robot picks a piece that does not exist",
        human=False, objects=[], coco_class=[], mixed=False,
        internal_cause="phantom pick (gripper error, unexpected idle)", window=(0.60, 1.00)),
    ("internal", "an4"): dict(
        track="internal", has_log=True, desc="Robot picks no piece at all",
        human=False, objects=[], coco_class=[], mixed=False,
        internal_cause="no pick (gripper error, unexpected idle)", window=(0.60, 1.00)),
    ("internal", "an5"): dict(
        track="internal", has_log=True, desc="Robot fails to grasp the piece",
        human=False, objects=[], coco_class=[], mixed=False,
        internal_cause="grasp failure (gripper error, unexpected idle)", window=(0.60, 1.00)),
    # --- no-log copies (an6=an1, an7=an2, an8=an3, an9=an4, an10=an5) ---
    ("internal", "an6"): dict(
        track="internal", has_log=False, desc="Robot picks a piece that does not exist (no log)",
        human=False, objects=[], coco_class=[], mixed=False,
        internal_cause="phantom pick (visual only: robot cycles but grips nothing)", window=(0.60, 1.00)),
    ("internal", "an7"): dict(
        track="internal", has_log=False, desc="Human hand rips a piece off the robot (mixed, no log)",
        human=True, objects=[], coco_class=[], mixed=True,
        internal_cause="human removes piece (visible in frame)", window=(0.25, 1.00)),
    ("internal", "an8"): dict(
        track="internal", has_log=False, desc="Robot picks a piece that does not exist (no log)",
        human=False, objects=[], coco_class=[], mixed=False,
        internal_cause="phantom pick (visual only: robot cycles but grips nothing)", window=(0.60, 1.00)),
    ("internal", "an9"): dict(
        track="internal", has_log=False, desc="Robot picks no piece at all (no log)",
        human=False, objects=[], coco_class=[], mixed=False,
        internal_cause="no pick (visual only: robot never closes on a piece)", window=(0.60, 1.00)),
    ("internal", "an10"): dict(
        track="internal", has_log=False, desc="Robot fails to grasp the piece (no log)",
        human=False, objects=[], coco_class=[], mixed=False,
        internal_cause="grasp failure (visual only: piece slips / not held)", window=(0.60, 1.00)),
}
# external entries predate the has_log field; default it in.
for _k, _g in GROUND_TRUTH.items():
    _g.setdefault("has_log", False)


def get(track, video):
    return GROUND_TRUTH.get((track, video))


def all_clips():
    return list(GROUND_TRUTH.items())


if __name__ == "__main__":
    for (tr, v), g in GROUND_TRUTH.items():
        tag = "MIXED" if g["mixed"] else g["track"]
        obj = ("; objects=" + ", ".join(f"{o}[{c or 'OOV'}]"
               for o, c in zip(g["objects"], g["coco_class"]))) if g["objects"] else ""
        print(f"{tr:9} {v:5} {tag:9} human={int(g['human'])} "
              f"win={g['window'][0]:.2f}-{g['window'][1]:.2f}{obj}")
        print(f"                     {g['desc']}")
