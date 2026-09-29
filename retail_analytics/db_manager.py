"""PostgreSQL Database Manager for Retail AI Analytics
=====================================================
Provides an asynchronous, non-blocking queue worker that writes analytics data
to PostgreSQL tables in real time without blocking the video tracking loop.

Supported Tables:
- customer_entries (track_id, timestamp, direction, age_group, gender, camera_id)
- customer_exits (track_id, timestamp, direction)
- store_occupancy (timestamp, camera_id, valid_entries, valid_exits, inside_visitors, interval_minutes)
- daily_metrics (date, in_count, out_count, person_with_bag_exit, demographics_count)
- customer_demographics (detailed VLM inferences)
"""

import queue
import threading
import time
from datetime import datetime, date
from typing import Optional, Dict, Any, Tuple

try:
    import psycopg2
    from psycopg2 import pool, extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from config.db_config import get_connection_params, ENABLE_POSTGRES, SHOP_ID
    from retail_analytics.utils import setup_logger
except ImportError:
    from utils import setup_logger
    ENABLE_POSTGRES = False
    SHOP_ID = 1

logger = setup_logger()


class DatabaseManager:
    """
    Thread-safe, non-blocking PostgreSQL Database Manager.
    Uses a background worker queue so video processing runs at 100% full speed.
    """

    def __init__(
        self,
        enable: bool = True,
        conn_params: Optional[Dict[str, Any]] = None,
        shop_id: Optional[int] = None,
    ):
        self.enabled = enable and ENABLE_POSTGRES and PSYCOPG2_AVAILABLE
        self.conn_params = conn_params or (get_connection_params() if ENABLE_POSTGRES else {})
        self.shop_id = shop_id if shop_id is not None else SHOP_ID
        self.job_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._conn = None
        self._worker_thread: Optional[threading.Thread] = None

        if not PSYCOPG2_AVAILABLE:
            logger.warning("[DatabaseManager] psycopg2 is not installed. Database writes disabled.")
            self.enabled = False
            return

        if not self.enabled:
            logger.info("[DatabaseManager] PostgreSQL integration is disabled by config.")
            return

        # Attempt initial connection
        self._connect()

        if self.is_connected:
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="PostgreSQL-WriterThread"
            )
            self._worker_thread.start()
            logger.info(
                f"[DatabaseManager] Connected to PostgreSQL ({self.conn_params.get('dbname')} "
                f"at {self.conn_params.get('host')}:{self.conn_params.get('port')}). "
                f"Background writer worker started."
            )
        else:
            logger.warning("[DatabaseManager] Initial connection failed. Worker will retry on queue submit.")

    @property
    def is_connected(self) -> bool:
        return self._conn is not None and not self._conn.closed

    def _connect(self) -> bool:
        """Establish or re-establish connection to PostgreSQL database."""
        try:
            if self._conn is not None and not self._conn.closed:
                self._conn.close()
            self._conn = psycopg2.connect(**self.conn_params)
            self._conn.autocommit = True
            return True
        except Exception as exc:
            logger.error(f"[DatabaseManager] Connection error: {exc}")
            self._conn = None
            return False

    def _worker_loop(self) -> None:
        """Background thread worker that drains the queue and executes SQL statements."""
        while not self._stop_event.is_set() or not self.job_queue.empty():
            try:
                sql, params = self.job_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Ensure active connection
            if not self.is_connected:
                reconnected = False
                for _ in range(3):
                    time.sleep(1.0)
                    if self._connect():
                        reconnected = True
                        break
                if not reconnected:
                    logger.error(f"[DatabaseManager] Could not reconnect. Dropping record: {sql[:40]}...")
                    self.job_queue.task_done()
                    continue

            try:
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
            except psycopg2.OperationalError as op_err:
                logger.warning(f"[DatabaseManager] Operational error ({op_err}). Reconnecting...")
                if self._connect():
                    try:
                        with self._conn.cursor() as cur:
                            cur.execute(sql, params)
                    except Exception as retry_err:
                        logger.error(f"[DatabaseManager] Retry failed: {retry_err}")
            except Exception as exc:
                logger.error(f"[DatabaseManager] Error executing SQL: {exc} | SQL: {sql[:80]} | Params: {params}")
            finally:
                self.job_queue.task_done()

    # -------------------------------------------------------------------------
    # Public Asynchronous Enqueue Methods
    # -------------------------------------------------------------------------

    def insert_entry(
        self,
        track_id: int,
        timestamp: Optional[str] = None,
        direction: str = "ENTRY",
        age_group: Optional[str] = None,
        gender: Optional[str] = None,
        camera_id: str = "cam_01",
        shop_id: Optional[int] = None,
    ) -> None:
        """Queue a validated entry record for insertion into customer_entries."""
        if not self.enabled:
            return

        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sid = shop_id if shop_id is not None else self.shop_id
        sql = """
            INSERT INTO customer_entries (track_id, timestamp, direction, age_group, gender, camera_id, shop_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        params = (track_id, ts, direction, age_group or "Unknown", gender or "Unknown", camera_id, sid)
        self.job_queue.put((sql, params))

    def insert_exit(
        self,
        track_id: int,
        timestamp: Optional[str] = None,
        direction: str = "EXIT",
        shop_id: Optional[int] = None,
    ) -> None:
        """Queue a validated exit record for insertion into customer_exits."""
        if not self.enabled:
            return

        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sid = shop_id if shop_id is not None else self.shop_id
        sql = """
            INSERT INTO customer_exits (track_id, timestamp, direction, shop_id)
            VALUES (%s, %s, %s, %s);
        """
        params = (track_id, ts, direction, sid)
        self.job_queue.put((sql, params))

    def insert_occupancy(
        self,
        timestamp: Optional[str] = None,
        camera_id: str = "cam_01",
        valid_entries: int = 0,
        valid_exits: int = 0,
        inside_visitors: Optional[int] = None,
        interval_minutes: float = 20.0,
        shop_id: Optional[int] = None,
    ) -> None:
        """Queue a periodic occupancy snapshot for insertion into store_occupancy."""
        if not self.enabled:
            return

        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        inside = inside_visitors if inside_visitors is not None else max(0, valid_entries - valid_exits)
        sid = shop_id if shop_id is not None else self.shop_id
        sql = """
            INSERT INTO store_occupancy (timestamp, camera_id, valid_entries, valid_exits, inside_visitors, interval_minutes, shop_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        params = (ts, camera_id, valid_entries, valid_exits, inside, float(interval_minutes), sid)
        self.job_queue.put((sql, params))

    def upsert_daily_metrics(
        self,
        date_val: Optional[str] = None,
        in_count: int = 0,
        out_count: int = 0,
        person_with_bag_exit: int = 0,
        demographics_count: int = 0,
        shop_id: Optional[int] = None,
    ) -> None:
        """Queue an upsert for daily_metrics table (inserts or updates current date summary)."""
        if not self.enabled:
            return

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
        self.job_queue.put((sql, params))

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
        """Queue a detailed VLM inference record for insertion into customer_demographics."""
        if not self.enabled or not result_dict:
            return

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
            result_dict.get("confidence", "Unknown"),
            sid,
        )
        self.job_queue.put((sql, params))

    def close(self, timeout: float = 10.0) -> None:
        """Drain the background queue and cleanly close the database connection."""
        if not self.enabled:
            return

        logger.info("[DatabaseManager] Flushing pending database queue...")
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        if self._conn and not self._conn.closed:
            try:
                self._conn.close()
                logger.info("[DatabaseManager] Database connection closed cleanly.")
            except Exception as exc:
                logger.error(f"[DatabaseManager] Error closing connection: {exc}")
