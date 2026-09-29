"""Asynchronous Non-Blocking PostgreSQL Database Manager
=====================================================
Targeted for low-resource edge deployment.
- Reads credentials from config/database.json.
- Detects missing/blank credentials cleanly without crashing.
- Decouples AI vision loop from database I/O via a bounded background worker queue.
- Reconnects automatically on network drops.
- Failure in PostgreSQL does NOT stall or crash video preview or tracking.
"""

import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import psycopg2
    from psycopg2 import extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger("DatabaseManager")


class DatabaseManager:
    """Thread-safe, non-blocking PostgreSQL writer for retail AI analytics."""

    def __init__(
        self,
        credentials_file: str = "config/database.json",
        shop_id: int = 1,
        camera_id: str = "cam_01",
        queue_size: int = 1000,
        enable: bool = True,
    ):
        self.credentials_file = credentials_file
        self.shop_id = shop_id
        self.camera_id = camera_id
        self.max_queue_size = queue_size
        self.job_queue: queue.Queue = queue.Queue(maxsize=self.max_queue_size)
        self._stop_event = threading.Event()
        self._conn = None
        self._worker_thread: Optional[threading.Thread] = None
        self.enabled = False
        self.credentials_configured = False
        self.conn_params: Dict[str, Any] = {}
        self._recent_entry_insertions: Dict[int, float] = {}
        self._recent_exit_insertions: Dict[int, float] = {}
        self._recent_demo_insertions: Dict[int, float] = {}

        if not enable:
            print("[DatabaseManager] Database persistence disabled by configuration.")
            return

        if not PSYCOPG2_AVAILABLE:
            print("[DatabaseManager] Warning: psycopg2 is not installed. Database writes disabled.")
            return

        # 1. Load credentials from file
        self._load_credentials()

        if not self.credentials_configured:
            print(
                f"[DatabaseManager] Notice: Database credentials in '{self.credentials_file}' "
                "are blank or incomplete. PostgreSQL persistence is disabled until configured."
            )
            return

        self.enabled = True

        # 2. Attempt initial connection
        connected = self._connect()
        if connected:
            print(
                f"[DatabaseManager] Connected to PostgreSQL "
                f"({self.conn_params.get('dbname')} at {self.conn_params.get('host')}:{self.conn_params.get('port')}). "
                "Background DB writer started."
            )
        else:
            print(
                f"[DatabaseManager] Initial connection failed to {self.conn_params.get('host')}. "
                "Worker will retry in the background without blocking video."
            )

        # 3. Start background worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="PostgreSQL-WriterThread",
        )
        self._worker_thread.start()

    def _load_credentials(self) -> None:
        """Parse config/database.json or fallback environment variables."""
        cfg_path = Path(self.credentials_file)
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    host = str(data.get("host", "")).strip()
                    port = str(data.get("port", "")).strip()
                    dbname = str(data.get("database", "")).strip()
                    user = str(data.get("user", "")).strip()
                    password = str(data.get("password", "")).strip()

                    # Blank credential check
                    if host and dbname and user:
                        self.conn_params = {
                            "host": host,
                            "port": int(port) if port.isdigit() else 5432,
                            "dbname": dbname,
                            "user": user,
                            "password": password,
                            "connect_timeout": 3,
                        }
                        self.credentials_configured = True
                        return
            except Exception as e:
                print(f"[DatabaseManager] Error reading {self.credentials_file}: {e}")

        # Fallback to environment variables if present
        env_host = os.getenv("LOCAL_DB_HOST")
        env_db = os.getenv("LOCAL_DB_NAME")
        env_user = os.getenv("LOCAL_DB_USER")
        if env_host and env_db and env_user:
            self.conn_params = {
                "host": env_host,
                "port": int(os.getenv("LOCAL_DB_PORT", "5432")),
                "dbname": env_db,
                "user": env_user,
                "password": os.getenv("LOCAL_DB_PASSWORD", ""),
                "connect_timeout": 3,
            }
            self.credentials_configured = True

    @property
    def is_connected(self) -> bool:
        return self._conn is not None and not self._conn.closed

    def _connect(self) -> bool:
        """Establish or re-establish a persistent PostgreSQL connection."""
        if not self.credentials_configured:
            return False
        try:
            if self._conn is not None and not self._conn.closed:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = psycopg2.connect(**self.conn_params)
            self._conn.autocommit = True
            return True
        except Exception as exc:
            self._conn = None
            return False

    def _worker_loop(self) -> None:
        """Background worker consuming SQL tasks from bounded queue."""
        while not self._stop_event.is_set() or not self.job_queue.empty():
            try:
                item = self.job_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            sql, params = item

            # Verify connection or attempt reconnection
            if not self.is_connected:
                reconnected = False
                for _ in range(3):
                    time.sleep(1.0)
                    if self._connect():
                        reconnected = True
                        break
                if not reconnected:
                    # Drop record if DB remains unreachable to prevent memory accumulation
                    self.job_queue.task_done()
                    continue

            try:
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
            except (psycopg2.OperationalError, psycopg2.DatabaseError) as op_err:
                print(f"[DatabaseManager] Database error ({op_err}). Reconnecting...")
                if self._connect():
                    try:
                        with self._conn.cursor() as cur:
                            cur.execute(sql, params)
                    except Exception as retry_err:
                        print(f"[DatabaseManager] Retry failed: {retry_err}")
            except Exception as exc:
                print(f"[DatabaseManager] Error executing SQL: {exc}")
            finally:
                self.job_queue.task_done()

    # -------------------------------------------------------------------------
    # Non-blocking Asynchronous Public Enqueue Methods
    # -------------------------------------------------------------------------

    def _enqueue(self, sql: str, params: Tuple) -> None:
        if not self.enabled:
            return
        try:
            self.job_queue.put_nowait((sql, params))
        except queue.Full:
            try:
                # Discard oldest task to bound queue size
                self.job_queue.get_nowait()
                self.job_queue.task_done()
                self.job_queue.put_nowait((sql, params))
            except Exception:
                pass

    def insert_entry(
        self,
        track_id: int,
        timestamp: Optional[str] = None,
        direction: str = "ENTRY",
        age_group: Optional[str] = None,
        gender: Optional[str] = None,
        camera_id: Optional[str] = None,
        shop_id: Optional[int] = None,
    ) -> None:
        """Record customer entry event in customer_entries table."""
        now_time = time.time()
        if track_id in self._recent_entry_insertions and (now_time - self._recent_entry_insertions[track_id]) < 120.0:
            return
        self._recent_entry_insertions[track_id] = now_time
        if len(self._recent_entry_insertions) > 500:
            for k in list(self._recent_entry_insertions.keys())[:-250]:
                self._recent_entry_insertions.pop(k, None)

        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cid = camera_id or self.camera_id
        sid = shop_id if shop_id is not None else self.shop_id
        sql = """
            INSERT INTO customer_entries (track_id, timestamp, direction, age_group, gender, camera_id, shop_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        params = (track_id, ts, direction, age_group or "Unknown", gender or "Unknown", cid, sid)
        self._enqueue(sql, params)

    def insert_exit(
        self,
        track_id: int,
        timestamp: Optional[str] = None,
        direction: str = "EXIT",
        shop_id: Optional[int] = None,
    ) -> None:
        """Record customer exit event in customer_exits table."""
        now_time = time.time()
        if track_id in self._recent_exit_insertions and (now_time - self._recent_exit_insertions[track_id]) < 60.0:
            return
        self._recent_exit_insertions[track_id] = now_time
        if len(self._recent_exit_insertions) > 500:
            for k in list(self._recent_exit_insertions.keys())[:-250]:
                self._recent_exit_insertions.pop(k, None)

        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sid = shop_id if shop_id is not None else self.shop_id
        sql = """
            INSERT INTO customer_exits (track_id, timestamp, direction, shop_id)
            VALUES (%s, %s, %s, %s);
        """
        params = (track_id, ts, direction, sid)
        self._enqueue(sql, params)

    def insert_occupancy(
        self,
        timestamp: Optional[str] = None,
        camera_id: Optional[str] = None,
        valid_entries: int = 0,
        valid_exits: int = 0,
        inside_visitors: Optional[int] = None,
        interval_minutes: float = 20.0,
        shop_id: Optional[int] = None,
    ) -> None:
        """Record periodic store occupancy snapshot."""
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cid = camera_id or self.camera_id
        inside = inside_visitors if inside_visitors is not None else max(0, valid_entries - valid_exits)
        sid = shop_id if shop_id is not None else self.shop_id
        sql = """
            INSERT INTO store_occupancy (timestamp, camera_id, valid_entries, valid_exits, inside_visitors, interval_minutes, shop_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        params = (ts, cid, valid_entries, valid_exits, inside, float(interval_minutes), sid)
        self._enqueue(sql, params)

    def upsert_daily_metrics(
        self,
        date_val: Optional[str] = None,
        in_count: int = 0,
        out_count: int = 0,
        person_with_bag_exit: int = 0,
        demographics_count: int = 0,
        shop_id: Optional[int] = None,
    ) -> None:
        """Update daily summary statistics in daily_metrics."""
        dt = date_val or date.today().strftime("%Y-%m-%d")
        sid = shop_id if shop_id is not None else self.shop_id
        sql = """
            INSERT INTO daily_metrics (date, in_count, out_count, person_with_bag_exit, demographics_count, shop_id, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (date) DO UPDATE SET
                in_count = EXCLUDED.in_count,
                out_count = EXCLUDED.out_count,
                person_with_bag_exit = EXCLUDED.person_with_bag_exit,
                demographics_count = EXCLUDED.demographics_count,
                shop_id = EXCLUDED.shop_id,
                last_updated = CURRENT_TIMESTAMP;
        """
        params = (dt, in_count, out_count, person_with_bag_exit, demographics_count, sid)
        self._enqueue(sql, params)

    def insert_demographic(
        self,
        track_id: int,
        timestamp: Optional[str] = None,
        frame_no: Optional[int] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        direction: str = "ENTRY",
        result_dict: Optional[Dict[str, Any]] = None,
        shop_id: Optional[int] = None,
    ) -> None:
        """Queue demographic result to customer_demographics table."""
        if not result_dict:
            return

        now_time = time.time()
        if track_id in self._recent_demo_insertions and (now_time - self._recent_demo_insertions[track_id]) < 120.0:
            return
        self._recent_demo_insertions[track_id] = now_time
        if len(self._recent_demo_insertions) > 500:
            for k in list(self._recent_demo_insertions.keys())[:-250]:
                self._recent_demo_insertions.pop(k, None)

        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dt = ts.split(" ")[0] if " " in ts else date.today().strftime("%Y-%m-%d")
        bx1, by1, bx2, by2 = bbox if bbox else (None, None, None, None)
        sid = shop_id if shop_id is not None else self.shop_id

        exact_age = result_dict.get("exact_age") or result_dict.get("age")
        try:
            exact_age_int = int(exact_age) if exact_age is not None else None
        except (ValueError, TypeError):
            exact_age_int = None

        sql = """
            INSERT INTO customer_demographics (
                track_id, date, timestamp, frame_no,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                direction, age_group, exact_age, gender, visibility, confidence, shop_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        params = (
            track_id,
            dt,
            ts,
            frame_no,
            bx1, by1, bx2, by2,
            direction,
            result_dict.get("age_group", result_dict.get("age", "Unknown")),
            exact_age_int,
            result_dict.get("gender", result_dict.get("apparent_gender", "Unknown")),
            result_dict.get("visibility", "Unknown"),
            str(result_dict.get("confidence", "Unknown")),
            sid,
        )
        self._enqueue(sql, params)

    def update_entry_demographics(
        self,
        track_id: int,
        age_group: str,
        gender: str,
    ) -> None:
        """Update customer_entries table with finalized demographic values."""
        sql = """
            UPDATE customer_entries
            SET age_group = %s, gender = %s
            WHERE track_id = %s AND (age_group = 'Unknown' OR gender = 'Unknown');
        """
        params = (age_group, gender, track_id)
        self._enqueue(sql, params)

    def close(self, timeout: float = 3.0) -> None:
        """Drain background queue and cleanly close connection."""
        if not self.enabled:
            return

        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        if self._conn and not self._conn.closed:
            try:
                self._conn.close()
            except Exception:
                pass
            print("[DatabaseManager] PostgreSQL connection closed.")
