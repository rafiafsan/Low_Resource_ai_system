import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

Point = Tuple[int, int]
GatePoints = Tuple[Point, Point]

PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
GATES_CONFIG_PATH = CONFIG_DIR / "gates.json"

LEGACY_FILES = {
    "footfall": CONFIG_DIR / "gate_config.json",
    "bag_exit": CONFIG_DIR / "gate_config_bag.json",
    "demographics": CONFIG_DIR / "age_gender_gate_config.json",
}

VALID_DETECTORS = ("master", "footfall", "bag_exit", "demographics")


class UnifiedGateManager:
    """Manages virtual calibration gates for all detectors (master, footfall, bag_exit, demographics).

    Reads and writes to config/gates.json, with automatic migration and backward-compatibility
    for legacy single-module configuration files.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or GATES_CONFIG_PATH
        self.gates: Dict[str, Dict[str, Any]] = {
            "master": {"gate_points": None, "inside_reference": None},
            "footfall": {"gate_points": None, "inside_reference": None},
            "bag_exit": {"gate_points": None, "inside_reference": None},
            "demographics": {"gate_points": None, "inside_reference": None},
        }
        self.load_config()

    def load_config(self) -> None:
        """Load gates from config/gates.json or migrate from legacy JSON files."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key in VALID_DETECTORS:
                        if key in data and isinstance(data[key], dict):
                            points = data[key].get("gate_points")
                            ref = data[key].get("inside_reference")
                            self.gates[key] = {
                                "gate_points": [tuple(p) for p in points] if points and len(points) == 2 else None,
                                "inside_reference": tuple(ref) if ref and len(ref) == 2 else None,
                            }
                return
            except Exception as e:
                print(f"[UnifiedGateManager] Warning: failed to load {self.config_path}: {e}. Trying legacy files...")

        # Fallback: Migrate from legacy files if gates.json does not exist
        self._migrate_legacy_files()

    def _migrate_legacy_files(self) -> None:
        """Read existing legacy JSON config files and populate gates."""
        migrated_any = False
        for detector, legacy_path in LEGACY_FILES.items():
            if legacy_path.exists():
                try:
                    with open(legacy_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        points = data.get("gate_points")
                        ref = data.get("inside_reference")
                        if points and len(points) == 2 and ref and len(ref) == 2:
                            self.gates[detector] = {
                                "gate_points": [tuple(p) for p in points],
                                "inside_reference": tuple(ref),
                            }
                            migrated_any = True
                except Exception:
                    pass

        # If footfall exists, use it as default master
        if self.gates["footfall"]["gate_points"] is not None:
            self.gates["master"] = dict(self.gates["footfall"])
        elif self.gates["bag_exit"]["gate_points"] is not None:
            self.gates["master"] = dict(self.gates["bag_exit"])
        elif self.gates["demographics"]["gate_points"] is not None:
            self.gates["master"] = dict(self.gates["demographics"])

        if migrated_any:
            print(f"[UnifiedGateManager] Migrated legacy gate files into {self.config_path}")
            self.save_config()

    def get_gate(self, detector: str = "master") -> Tuple[Optional[GatePoints], Optional[Point]]:
        """Retrieve (gate_points, inside_reference) for a detector.

        Falls back to 'master' if the specific detector gate is not set.
        """
        gate = self.gates.get(detector)
        if gate and gate["gate_points"] and gate["inside_reference"]:
            pts = (tuple(gate["gate_points"][0]), tuple(gate["gate_points"][1]))
            ref = tuple(gate["inside_reference"])
            return pts, ref

        # Fallback to master
        master = self.gates.get("master", {})
        if master.get("gate_points") and master.get("inside_reference"):
            pts = (tuple(master["gate_points"][0]), tuple(master["gate_points"][1]))
            ref = tuple(master["inside_reference"])
            return pts, ref

        # Fallback to any configured gate
        for name in ("footfall", "bag_exit", "demographics"):
            g = self.gates.get(name, {})
            if g.get("gate_points") and g.get("inside_reference"):
                pts = (tuple(g["gate_points"][0]), tuple(g["gate_points"][1]))
                ref = tuple(g["inside_reference"])
                return pts, ref

        return None, None

    def set_gate(
        self,
        detector: str,
        gate_points: Optional[List[Point]],
        inside_reference: Optional[Point],
    ) -> None:
        """Set gate configuration for a specific detector."""
        if detector not in VALID_DETECTORS:
            raise ValueError(f"Invalid detector: {detector}. Must be one of {VALID_DETECTORS}")

        self.gates[detector] = {
            "gate_points": [tuple(p) for p in gate_points] if gate_points and len(gate_points) == 2 else None,
            "inside_reference": tuple(inside_reference) if inside_reference and len(inside_reference) == 2 else None,
        }

    def has_gate(self, detector: str = "master") -> bool:
        """Check if a valid gate exists for the detector (or via master fallback)."""
        pts, ref = self.get_gate(detector)
        return pts is not None and ref is not None

    def save_config(self) -> None:
        """Save all gate configurations to config/gates.json and sync legacy files."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.gates, f, indent=4)

        # Sync back to legacy files for backward compatibility
        self._sync_legacy_files()

    def _sync_legacy_files(self) -> None:
        """Keep legacy JSON files synchronized so older scripts don't break."""
        for detector, legacy_path in LEGACY_FILES.items():
            pts, ref = self.get_gate(detector)
            if pts and ref:
                try:
                    with open(legacy_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "gate_points": [list(pts[0]), list(pts[1])],
                                "inside_reference": list(ref),
                            },
                            f,
                            indent=4,
                        )
                except Exception:
                    pass
