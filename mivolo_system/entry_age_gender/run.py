"""Print age and gender only for people crossing into the configured gate.

The detector runs on every frame to retain stable tracking IDs. MiVOLO age/gender
inference is invoked only for a person whose track moves from OUTSIDE to INSIDE.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import torch


MIVOLO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MIVOLO_ROOT.parent
GATE_CONFIG_PATH = PROJECT_ROOT / "config" / "age_gender_gate_config.json"
sys.path.insert(0, str(MIVOLO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from gate.geometry import get_side  # noqa: E402


Point = Tuple[int, int]
MAX_MISSING_FRAMES = 15
AGE_GENDER_START_FRAME = 15


@dataclass(frozen=True)
class GateConfig:
    gate_points: Optional[Tuple[Point, Point]] = None
    inside_reference: Optional[Point] = None

    @property
    def is_complete(self) -> bool:
        return self.gate_points is not None and self.inside_reference is not None

    @classmethod
    def load(cls, path: Path) -> "GateConfig":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in gate configuration: {path}") from exc

        gate_points = data.get("gate_points", [])
        inside_reference = data.get("inside_reference")
        if not gate_points and inside_reference is None:
            return cls()

        if len(gate_points) != 2 or len(inside_reference) != 2 or any(len(point) != 2 for point in gate_points):
            raise RuntimeError(f"Invalid gate geometry in: {path}")

        return cls(
            gate_points=(tuple(gate_points[0]), tuple(gate_points[1])),
            inside_reference=tuple(inside_reference),
        )

    def save(self, path: Path) -> None:
        if not self.is_complete:
            raise RuntimeError("Cannot save an incomplete gate configuration.")
        path.write_text(
            json.dumps(
                {
                    "gate_points": self.gate_points,
                    "inside_reference": self.inside_reference,
                },
                indent=4,
            ),
            encoding="utf-8",
        )


class EntryGateMonitor:
    """Identify tracks that begin outside and later cross into the shop."""

    def __init__(self, gate_config: GateConfig):
        self.gate_config = gate_config
        self.previous_sides: Dict[int, str] = {}
        self.started_outside: set[int] = set()
        self.reported_entries: set[int] = set()

    def entering_track_ids(self, persons: Iterable[Tuple[int, Point]]) -> List[int]:
        assert self.gate_config.gate_points is not None
        assert self.gate_config.inside_reference is not None
        entering_ids: List[int] = []
        for track_id, point in persons:
            current_side = get_side(
                point,
                self.gate_config.gate_points[0],
                self.gate_config.gate_points[1],
                self.gate_config.inside_reference,
            )
            if current_side == "GATE":
                continue

            previous_side = self.previous_sides.get(track_id)
            self.previous_sides[track_id] = current_side
            if current_side == "OUTSIDE":
                self.started_outside.add(track_id)
            if (
                previous_side == "OUTSIDE"
                and current_side == "INSIDE"
                and track_id in self.started_outside
                and track_id not in self.reported_entries
            ):
                self.reported_entries.add(track_id)
                entering_ids.append(track_id)
        return entering_ids


@dataclass
class TrackEvidence:
    """Per-frame MiVOLO observations retained in memory for one tracked person."""

    observations: List[Tuple[Optional[float], Optional[str]]]
    missing_frames: int = 0

    def record(self, age: Optional[float], gender: Optional[str]) -> None:
        self.observations.append((age, gender))
        self.missing_frames = 0

    @property
    def average_age(self) -> Optional[float]:
        ages = [age for age, _ in self.observations if age is not None]
        return sum(ages) / len(ages) if ages else None

    @property
    def gender_votes(self) -> Counter:
        return Counter(gender.lower() for _, gender in self.observations if gender is not None)

    @property
    def final_gender(self) -> Optional[str]:
        votes = self.gender_votes
        if not votes:
            return None
        # Counter returns the most frequent gender; ties preserve first detection.
        return votes.most_common(1)[0][0]


class EntryMiVOLOConfig:
    def __init__(self, device: str):
        self.detector_weights = str(MIVOLO_ROOT / "model" / "yolov8x_person_face.pt")
        self.checkpoint = str(MIVOLO_ROOT / "model" / "model_imdb_cross_person_4.22_99.46.pth.tar")
        self.device = device
        self.with_persons = True
        self.disable_faces = False
        self.draw = False


def get_person_tracks(detected_objects: PersonAndFaceResult) -> List[Tuple[int, Point, int]]:
    """Return (track ID, gate point, YOLO result index) for tracked persons."""
    tracks: List[Tuple[int, Point, int]] = []
    boxes = detected_objects.yolo_results.boxes
    for result_index in detected_objects.get_bboxes_inds("person"):
        box = boxes[result_index]
        if box.id is None:
            continue
        x1, y1, x2, _ = box.xyxy.squeeze().cpu().tolist()
        track_id = int(box.id.item())
        gate_point = (int((x1 + x2) / 2), int(y1))
        tracks.append((track_id, gate_point, result_index))
    return tracks


def infer_people(
    predictor: Predictor,
    detected_objects: PersonAndFaceResult,
    track_ids: List[int],
    person_tracks: List[Tuple[int, Point, int]],
) -> Dict[int, Tuple[Optional[float], Optional[str]]]:
    """Run MiVOLO only for the specified tracked people in the current frame."""
    from MiVOLO.structures import PersonAndFaceResult

    selected_indices = [index for track_id, _, index in person_tracks if track_id in track_ids]
    if not selected_indices:
        return {}

    # Results supports indexing and retains the original image and persistent
    # tracking IDs. Limiting it here prevents age/gender inference for leavers
    # and other people visible in the same frame.
    selected_results = PersonAndFaceResult(detected_objects.yolo_results[selected_indices])
    predictor.age_gender_model.predict(detected_objects.yolo_results.orig_img, selected_results)
    persons, _ = selected_results.get_results_for_tracking()
    return {track_id: persons.get(track_id, (None, None)) for track_id in track_ids}


def draw_gate(frame, gate_config: GateConfig) -> None:
    assert gate_config.gate_points is not None
    assert gate_config.inside_reference is not None
    start, end = gate_config.gate_points
    cv2.line(frame, start, end, (0, 255, 255), 2)
    cv2.circle(frame, gate_config.inside_reference, 5, (0, 255, 0), -1)
    cv2.putText(frame, "INSIDE", gate_config.inside_reference, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def configure_gate(source) -> GateConfig:
    """Collect two gate points and one inside reference point on the first frame."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source for gate setup: {source}")
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not read a frame for gate setup.")

    window_name = "Configure Age/Gender Gate"
    selected_points: List[Point] = []
    inside_reference: Optional[Point] = None

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal inside_reference
        if event != cv2.EVENT_LBUTTONDOWN or inside_reference is not None:
            return
        if len(selected_points) < 2:
            selected_points.append((x, y))
        else:
            inside_reference = (x, y)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    print("Configure age/gender gate: click two gate endpoints, then click inside the shop.")
    try:
        while inside_reference is None:
            canvas = frame.copy()
            for point in selected_points:
                cv2.circle(canvas, point, 5, (0, 255, 255), -1)
            if len(selected_points) == 2:
                cv2.line(canvas, selected_points[0], selected_points[1], (0, 255, 255), 2)
            cv2.putText(
                canvas,
                "Click: gate start, gate end, then INSIDE point | r: reset | q: cancel",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                selected_points.clear()
            elif key == ord("q"):
                raise RuntimeError("Gate setup cancelled. No age/gender inference was started.")
    finally:
        cv2.destroyWindow(window_name)

    gate_config = GateConfig(gate_points=(selected_points[0], selected_points[1]), inside_reference=inside_reference)
    gate_config.save(GATE_CONFIG_PATH)
    print(f"Saved age/gender gate configuration to: {GATE_CONFIG_PATH}")
    return gate_config


def draw_entry_annotations(
    frame,
    detected_objects: PersonAndFaceResult,
    person_tracks: List[Tuple[int, Point, int]],
    evidence_by_track: Dict[int, TrackEvidence],
    frame_genders: Dict[int, Optional[str]],
) -> None:
    """Draw current-frame gender and accumulated evidence for monitored tracks."""
    for track_id, _, result_index in person_tracks:
        if track_id not in evidence_by_track:
            continue

        x1, y1, x2, y2 = map(int, detected_objects.yolo_results.boxes[result_index].xyxy.squeeze().cpu().tolist())
        evidence = evidence_by_track[track_id]
        age = evidence.average_age
        gender = frame_genders.get(track_id)
        label = (
            f"ID {track_id} | Age avg {age:.1f} | Gender {gender.title()} | Frames {len(evidence.observations)}"
            if age is not None and gender is not None
            else f"ID {track_id} | Age avg {age:.1f} | Gender unavailable | Frames {len(evidence.observations)}"
            if age is not None
            else f"ID {track_id} | Age unavailable | Gender {gender.title()} | Frames {len(evidence.observations)}"
            if gender is not None
            else f"ID {track_id} | Age/Gender unavailable"
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def parse_args() -> argparse.Namespace:
    from config.settings import VIDEO_SOURCE

    parser = argparse.ArgumentParser(description="Detect age and gender for shop entries only.")
    parser.add_argument("--source", default=VIDEO_SOURCE, help="Video file, stream URL, or camera index.")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV preview window.")
    return parser.parse_args()


def resolve_source(source: str):
    return int(source) if source.isdigit() else source


def print_final_result(track_id: int, evidence: TrackEvidence) -> None:
    """Print one final result after all available frame observations are counted."""
    age = evidence.average_age
    votes = evidence.gender_votes
    final_gender = evidence.final_gender
    age_text = f"{age:.1f}" if age is not None else "unavailable"
    gender_text = final_gender.title() if final_gender is not None else "unavailable"
    print(
        f"ENTRY FINAL | track_id={track_id} | frames={len(evidence.observations)} "
        f"| age_average={age_text} | gender={gender_text} "
        f"| male_votes={votes.get('male', 0)} | female_votes={votes.get('female', 0)}"
    )


def finalize_missing_tracks(
    evidence_by_track: Dict[int, TrackEvidence],
    visible_track_ids: set[int],
    entered_track_ids: set[int],
) -> None:
    """Finalize entered tracks after a short absence; discard non-entry tracks."""
    for track_id, evidence in list(evidence_by_track.items()):
        if track_id in visible_track_ids:
            continue
        evidence.missing_frames += 1
        if evidence.missing_frames < MAX_MISSING_FRAMES:
            continue
        if track_id in entered_track_ids:
            print_final_result(track_id, evidence)
        del evidence_by_track[track_id]


def main() -> None:
    args = parse_args()
    source = resolve_source(args.source)
    gate_config = GateConfig.load(GATE_CONFIG_PATH)
    if not gate_config.is_complete:
        if args.no_display:
            raise RuntimeError("Gate setup needs a display. Run once without --no-display to create the configuration.")
        gate_config = configure_gate(source)
    from MiVOLO.predictor import Predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = Predictor(EntryMiVOLOConfig(device), verbose=False)
    gate_monitor = EntryGateMonitor(gate_config)
    evidence_by_track: Dict[int, TrackEvidence] = {}
    track_frame_counts: Dict[int, int] = {}

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    print(f"Using gate configuration: {GATE_CONFIG_PATH}")
    print(f"Processing {source} on {device}. Frame observations begin outside the gate.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detected_objects = predictor.detector.track(frame)
            person_tracks = get_person_tracks(detected_objects)
            entering_ids = gate_monitor.entering_track_ids(
                (track_id, point) for track_id, point, _ in person_tracks
            )
            monitored_track_ids = [
                track_id for track_id, _, _ in person_tracks if track_id in gate_monitor.started_outside
            ]
            for track_id in monitored_track_ids:
                track_frame_counts[track_id] = track_frame_counts.get(track_id, 0) + 1
            ready_track_ids = [
                track_id
                for track_id in monitored_track_ids
                if track_frame_counts[track_id] >= AGE_GENDER_START_FRAME
            ]
            frame_results = infer_people(predictor, detected_objects, ready_track_ids, person_tracks)

            for track_id, result in frame_results.items():
                evidence_by_track.setdefault(track_id, TrackEvidence(observations=[])).record(*result)

            frame_genders = {track_id: gender for track_id, (_, gender) in frame_results.items()}
            visible_track_ids = {track_id for track_id, _, _ in person_tracks}
            finalize_missing_tracks(evidence_by_track, visible_track_ids, gate_monitor.reported_entries)

            if not args.no_display:
                draw_gate(frame, gate_config)
                draw_entry_annotations(frame, detected_objects, person_tracks, evidence_by_track, frame_genders)
                cv2.imshow("Entry-only Age and Gender", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        for track_id, evidence in list(evidence_by_track.items()):
            if track_id in gate_monitor.reported_entries:
                print_final_result(track_id, evidence)
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
