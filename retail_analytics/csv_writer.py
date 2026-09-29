"""
csv_writer.py — Persistent CSV storage for retail analytics.

Manages CSV tables for retail analytics:
1. EntryWriter (output/entries.csv):
   Stores validated customer entries with demographic profiling:
   - id: Tracker person ID
   - timestamp: Wall-clock timestamp (YYYY-MM-DD HH:MM:SS)
   - direction: ENTRY
   - age_group: Child | Teenager | Young Adult | Adult | Senior
   - gender: Male | Female | Uncertain
   - camera_id: Camera identifier (e.g. cam_01)

2. ExitWriter (output/exits.csv):
   Stores confirmed customer exits (validated after dwell threshold):
   - id: Tracker person ID
   - timestamp: Wall-clock timestamp (YYYY-MM-DD HH:MM:SS)
   - direction: EXIT
   - camera_id: Camera identifier (e.g. cam_01)

3. OccupancyWriter (output/occupancy_20min.csv):
   Logs periodic visitor occupancy inside the store every 20 minutes:
   - timestamp: Log timestamp (YYYY-MM-DD HH:MM:SS)
   - camera_id: Camera identifier (e.g. cam_01)
   - valid_entries: Cumulative validated entries
   - valid_exits: Cumulative validated exits
   - inside_visitors: max(0, valid_entries - valid_exits)
   - interval_minutes: Periodic interval duration in minutes (default 20.0)

4. DailyMetricsWriter (output/daily_metrics.csv):
   Daily summary metrics auto-incrementing across days.

5. CSVWriter (output/customer_demographics.csv):
   Full demographic records per validated entering customer.
"""

import csv
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

try:
    from retail_analytics.utils import setup_logger, ensure_dir
except ImportError:
    from utils import setup_logger, ensure_dir

logger = setup_logger()

# ---------------------------------------------------------------------------
# CSV Column Definitions
# ---------------------------------------------------------------------------
ENTRY_COLUMNS = [
    "id",
    "timestamp",
    "direction",
    "age_group",
    "gender",
    "camera_id",
]

EXIT_COLUMNS = [
    "id",
    "timestamp",
    "direction",
]

OCCUPANCY_COLUMNS = [
    "timestamp",
    "camera_id",
    "valid_entries",
    "valid_exits",
    "inside_visitors",
    "interval_minutes",
]

DEMOGRAPHICS_COLUMNS = [
    "track_id",
    "date",
    "timestamp",
    "frame_no",
    "age_group",
    "gender",
    "confidence",
    "direction",
    "bbox",
]

DAILY_METRICS_COLUMNS = [
    "date",
    "in_count",
    "out_count",
    "bag_exit_count",
    "demographics_count",
    "last_updated",
]


# ---------------------------------------------------------------------------
# 1. EntryWriter
# ---------------------------------------------------------------------------

class EntryWriter:
    """
    Thread-safe CSV writer for validated store entries with demographic profiles.
    Output table: id, timestamp, direction, age_group, gender, camera_id
    """

    def __init__(self, output_path: str = "output/entries.csv"):
        self._path = Path(output_path)
        self._lock = threading.Lock()
        ensure_dir(str(self._path.parent))
        self._ensure_header()
        logger.info(f"[EntryWriter] Writing entries to: {self._path.resolve()}")

    def write_entry(
        self,
        id: int,
        timestamp: Optional[str] = None,
        direction: str = "ENTRY",
        age_group: str = "Adult",
        gender: str = "Unknown",
        camera_id: str = "cam_01",
        result_dict: Optional[dict] = None,
    ) -> None:
        """Append one validated entry record to entries.csv."""
        if result_dict:
            age_group = result_dict.get("age_group", age_group or "Adult")
            gender = result_dict.get(
                "gender", result_dict.get("apparent_gender", gender or "Unknown")
            )

        ts_str = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gender_str = gender.title() if isinstance(gender, str) else str(gender)

        row = {
            "id": id,
            "timestamp": ts_str,
            "direction": direction,
            "age_group": age_group,
            "gender": gender_str,
            "camera_id": camera_id,
        }

        with self._lock:
            try:
                with open(self._path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=ENTRY_COLUMNS)
                    writer.writerow(row)
                logger.info(
                    f"[EntryWriter] Saved entry: id={id}, dir={direction}, "
                    f"gender={gender_str}, age={age_group}, cam={camera_id}"
                )
            except Exception as exc:
                logger.error(f"[EntryWriter] Failed to write entry row for id {id}: {exc}")

    def _ensure_header(self) -> None:
        needs_header = not self._path.exists() or self._path.stat().st_size == 0
        if needs_header:
            try:
                with open(self._path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=ENTRY_COLUMNS)
                    writer.writeheader()
                logger.info(f"[EntryWriter] Created new entries CSV with header: {self._path}")
            except Exception as exc:
                logger.error(f"[EntryWriter] Failed to create CSV header: {exc}")

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# 2. ExitWriter
# ---------------------------------------------------------------------------

class ExitWriter:
    """
    Thread-safe CSV writer for validated store exits.
    Output table: id, timestamp, direction, camera_id
    """

    def __init__(self, output_path: str = "output/exits.csv"):
        self._path = Path(output_path)
        self._lock = threading.Lock()
        ensure_dir(str(self._path.parent))
        self._ensure_header()
        logger.info(f"[ExitWriter] Writing exits to: {self._path.resolve()}")

    def write_exit(
        self,
        id: int,
        timestamp: Optional[str] = None,
        direction: str = "EXIT",
        camera_id: Optional[str] = None,
    ) -> None:
        """Append one validated exit record to exits.csv."""
        ts_str = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "id": id,
            "timestamp": ts_str,
            "direction": direction,
        }

        with self._lock:
            try:
                with open(self._path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=EXIT_COLUMNS)
                    writer.writerow(row)
                logger.info(f"[ExitWriter] Saved exit: id={id}, dir={direction}")
            except Exception as exc:
                logger.error(f"[ExitWriter] Failed to write exit row for id {id}: {exc}")

    def _ensure_header(self) -> None:
        needs_header = not self._path.exists() or self._path.stat().st_size == 0
        if needs_header:
            try:
                with open(self._path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=EXIT_COLUMNS)
                    writer.writeheader()
                logger.info(f"[ExitWriter] Created new exits CSV with header: {self._path}")
            except Exception as exc:
                logger.error(f"[ExitWriter] Failed to create CSV header: {exc}")

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# 3. OccupancyWriter (Every 20 Minutes)
# ---------------------------------------------------------------------------

class OccupancyWriter:
    """
    Thread-safe CSV writer for store visitor occupancy logged periodically (e.g. every 20 min).
    Formula: inside_visitors = max(0, valid_entries - valid_exits)
    """

    def __init__(self, output_path: str = "output/occupancy_20min.csv"):
        self._path = Path(output_path)
        self._lock = threading.Lock()
        ensure_dir(str(self._path.parent))
        self._ensure_header()
        logger.info(f"[OccupancyWriter] Writing occupancy logs to: {self._path.resolve()}")

    def log_occupancy(
        self,
        valid_entries: int,
        valid_exits: int,
        camera_id: str = "cam_01",
        timestamp: Optional[str] = None,
        interval_minutes: float = 20.0,
    ) -> int:
        """
        Compute inside visitors as max(0, valid_entries - valid_exits) and log row.
        Returns the computed inside_visitors count.
        """
        inside_visitors = max(0, valid_entries - valid_exits)
        ts_str = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "timestamp": ts_str,
            "camera_id": camera_id,
            "valid_entries": valid_entries,
            "valid_exits": valid_exits,
            "inside_visitors": inside_visitors,
            "interval_minutes": interval_minutes,
        }

        with self._lock:
            try:
                with open(self._path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=OCCUPANCY_COLUMNS)
                    writer.writerow(row)
                logger.info(
                    f"[OccupancyWriter] Logged occupancy: in={valid_entries}, out={valid_exits}, "
                    f"inside={inside_visitors}, cam={camera_id}, interval={interval_minutes}m"
                )
            except Exception as exc:
                logger.error(f"[OccupancyWriter] Failed to write occupancy record: {exc}")

        return inside_visitors

    def _ensure_header(self) -> None:
        needs_header = not self._path.exists() or self._path.stat().st_size == 0
        if needs_header:
            try:
                with open(self._path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=OCCUPANCY_COLUMNS)
                    writer.writeheader()
                logger.info(f"[OccupancyWriter] Created new occupancy CSV with header: {self._path}")
            except Exception as exc:
                logger.error(f"[OccupancyWriter] Failed to create CSV header: {exc}")

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# 4. CSVWriter (Customer Demographics)
# ---------------------------------------------------------------------------

class CSVWriter:
    """
    Thread-safe CSV writer for per-track demographic analysis results.
    Filters to only record customers entering into the shop.
    """

    def __init__(self, output_path: str = "output/customer_demographics.csv"):
        self._path = Path(output_path)
        self._lock = threading.Lock()

        ensure_dir(str(self._path.parent))
        self._ensure_header()

        logger.info(f"[CSVWriter] Writing customer demographics to: {self._path.resolve()}")

    def write_result(
        self,
        track_id: int,
        timestamp: str,
        frame_no: int,
        bbox: tuple,
        age_group: str = "",
        gender: str = "",
        confidence: str = "",
        direction: str = "ENTRY",
        result_dict: Optional[dict] = None,
    ) -> None:
        """Append one demographic result row to the CSV file."""
        if result_dict:
            age_group = result_dict.get("age_group", age_group or "Adult")
            gender = result_dict.get(
                "gender", result_dict.get("apparent_gender", gender or "Unknown")
            )
            confidence = result_dict.get("confidence", confidence or "High")

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        ts_str = timestamp or now.strftime("%Y-%m-%d %H:%M:%S")
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}" if bbox else ""

        row = {
            "track_id": track_id,
            "date": date_str,
            "timestamp": ts_str,
            "frame_no": frame_no,
            "age_group": age_group,
            "gender": gender.title() if isinstance(gender, str) else str(gender),
            "confidence": confidence,
            "direction": direction,
            "bbox": bbox_str,
        }

        with self._lock:
            try:
                with open(self._path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=DEMOGRAPHICS_COLUMNS)
                    writer.writerow(row)
                logger.info(
                    f"[CSVWriter] Saved demographic record: track={track_id}, "
                    f"gender={gender}, age={age_group}, direction={direction}"
                )
            except Exception as exc:
                logger.error(f"[CSVWriter] Failed to write row for track {track_id}: {exc}")

    def _ensure_header(self) -> None:
        needs_header = not self._path.exists() or self._path.stat().st_size == 0
        if needs_header:
            try:
                with open(self._path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=DEMOGRAPHICS_COLUMNS)
                    writer.writeheader()
                logger.info(f"[CSVWriter] Created new CSV with header: {self._path}")
            except Exception as exc:
                logger.error(f"[CSVWriter] Failed to create CSV header: {exc}")

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# 5. DailyMetricsWriter (Cumulative Store Totals)
# ---------------------------------------------------------------------------

class DailyMetricsWriter:
    """
    Thread-safe CSV writer for cumulative daily retail store metrics.
    Maintains one row per date in output/daily_metrics.csv.
    Auto-increments and updates today's record across sessions and runs.
    """

    def __init__(self, output_path: str = "output/daily_metrics.csv"):
        self._path = Path(output_path)
        self._lock = threading.Lock()

        ensure_dir(str(self._path.parent))
        self._ensure_header()

        self.today_str = datetime.now().strftime("%Y-%m-%d")
        self.in_count: int = 0
        self.out_count: int = 0
        self.bag_exit_count: int = 0
        self.demographics_count: int = 0

        self._load_today_baseline()
        logger.info(
            f"[DailyMetricsWriter] Initialized for date {self.today_str}: "
            f"in={self.in_count}, out={self.out_count}, "
            f"bag_exit={self.bag_exit_count}, demographics={self.demographics_count}"
        )

    def _ensure_header(self) -> None:
        needs_header = not self._path.exists() or self._path.stat().st_size == 0
        if needs_header:
            try:
                with open(self._path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=DAILY_METRICS_COLUMNS)
                    writer.writeheader()
                logger.info(f"[DailyMetricsWriter] Created daily metrics CSV: {self._path}")
            except Exception as exc:
                logger.error(f"[DailyMetricsWriter] Failed to create header: {exc}")

    def _load_today_baseline(self) -> None:
        if not self._path.exists():
            return

        with self._lock:
            try:
                with open(self._path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("date") == self.today_str:
                            self.in_count = int(row.get("in_count", 0) or 0)
                            self.out_count = int(row.get("out_count", 0) or 0)
                            self.bag_exit_count = int(row.get("bag_exit_count", 0) or 0)
                            self.demographics_count = int(row.get("demographics_count", 0) or 0)
            except Exception as exc:
                logger.warning(f"[DailyMetricsWriter] Could not load today's baseline: {exc}")

    def update_session_counts(
        self,
        session_in: int,
        session_out: int,
        session_bag_exit: int,
        session_demo: int,
    ) -> None:
        now = datetime.now()
        current_date_str = now.strftime("%Y-%m-%d")

        with self._lock:
            if current_date_str != self.today_str:
                self.today_str = current_date_str
                self.in_count = 0
                self.out_count = 0
                self.bag_exit_count = 0
                self.demographics_count = 0

            self.in_count = session_in
            self.out_count = session_out
            self.bag_exit_count = session_bag_exit
            self.demographics_count = session_demo

            self._flush_to_csv(now.strftime("%Y-%m-%d %H:%M:%S"))

    def increment_metric(self, metric: str, amount: int = 1) -> None:
        now = datetime.now()
        current_date_str = now.strftime("%Y-%m-%d")

        with self._lock:
            if current_date_str != self.today_str:
                self.today_str = current_date_str
                self.in_count = 0
                self.out_count = 0
                self.bag_exit_count = 0
                self.demographics_count = 0

            if metric == "in_count":
                self.in_count += amount
            elif metric == "out_count":
                self.out_count += amount
            elif metric == "bag_exit_count":
                self.bag_exit_count += amount
            elif metric == "demographics_count":
                self.demographics_count += amount

            self._flush_to_csv(now.strftime("%Y-%m-%d %H:%M:%S"))

    def _flush_to_csv(self, timestamp_str: str) -> None:
        rows: List[Dict[str, Any]] = []
        found = False

        try:
            if self._path.exists() and self._path.stat().st_size > 0:
                with open(self._path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("date") == self.today_str:
                            row["in_count"] = self.in_count
                            row["out_count"] = self.out_count
                            row["bag_exit_count"] = self.bag_exit_count
                            row["demographics_count"] = self.demographics_count
                            row["last_updated"] = timestamp_str
                            found = True
                        rows.append(row)

            if not found:
                rows.append({
                    "date": self.today_str,
                    "in_count": self.in_count,
                    "out_count": self.out_count,
                    "bag_exit_count": self.bag_exit_count,
                    "demographics_count": self.demographics_count,
                    "last_updated": timestamp_str,
                })

            temp_path = self._path.with_suffix(".tmp")
            with open(temp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=DAILY_METRICS_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            temp_path.replace(self._path)
        except Exception as exc:
            logger.error(f"[DailyMetricsWriter] Failed to flush metrics to CSV: {exc}")

    @property
    def path(self) -> Path:
        return self._path
