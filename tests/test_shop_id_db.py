"""Verification script for shop_id integration in DatabaseManager and PostgreSQL tables."""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from config.db_config import get_connection_params, SHOP_ID
from retail_analytics.db_manager import DatabaseManager

def test_db_manager_shop_id():
    print(f"Testing DatabaseManager with default SHOP_ID={SHOP_ID}...")
    db = DatabaseManager(enable=True, shop_id=SHOP_ID)
    assert db.enabled, "DatabaseManager should be enabled"
    assert db.is_connected, "DatabaseManager should be connected"

    test_track_id = 999999
    test_cam = "test_cam_shop_id"
    test_ts = "2026-09-22 12:00:00"

    print("Queuing test records across all tables...")
    # 1. Entry
    db.insert_entry(
        track_id=test_track_id,
        timestamp=test_ts,
        direction="ENTRY",
        age_group="20-30",
        gender="Male",
        camera_id=test_cam,
        shop_id=1,
    )

    # 2. Exit
    db.insert_exit(
        track_id=test_track_id,
        timestamp=test_ts,
        direction="EXIT",
        shop_id=1,
    )

    # 3. Occupancy
    db.insert_occupancy(
        camera_id=test_cam,
        valid_entries=10,
        valid_exits=4,
        interval_minutes=20.0,
        shop_id=1,
    )

    # 4. Daily Metrics
    db.upsert_daily_metrics(
        in_count=10,
        out_count=4,
        person_with_bag_exit=2,
        demographics_count=8,
        shop_id=1,
    )

    # 5. Demographic
    db.insert_demographic(
        track_id=test_track_id,
        timestamp=test_ts,
        frame_no=100,
        bbox=[10, 20, 100, 200],
        direction="ENTRY",
        result_dict={"age_group": "20-30", "gender": "Male", "confidence": 0.95},
        shop_id=1,
    )

    print("Waiting for queue worker to flush...")
    db.close(timeout=5.0)

    print("Connecting directly to PostgreSQL to verify inserted shop_id columns...")
    conn = psycopg2.connect(**get_connection_params())
    conn.autocommit = True
    cur = conn.cursor()

    try:
        # Check customer_entries
        cur.execute("SELECT entry_id, track_id, shop_id, camera_id FROM customer_entries WHERE track_id = %s", (test_track_id,))
        rows = cur.fetchall()
        print(f"customer_entries rows: {rows}")
        assert len(rows) > 0, "No row found in customer_entries"
        assert rows[0][2] == 1, f"Expected shop_id=1, got {rows[0][2]}"

        # Check customer_exits
        cur.execute("SELECT exit_id, track_id, shop_id FROM customer_exits WHERE track_id = %s", (test_track_id,))
        rows = cur.fetchall()
        print(f"customer_exits rows: {rows}")
        assert len(rows) > 0, "No row found in customer_exits"
        assert rows[0][2] == 1, f"Expected shop_id=1, got {rows[0][2]}"

        # Check store_occupancy
        cur.execute("SELECT occupancy_id, camera_id, shop_id FROM store_occupancy WHERE camera_id = %s", (test_cam,))
        rows = cur.fetchall()
        print(f"store_occupancy rows: {rows}")
        assert len(rows) > 0, "No row found in store_occupancy"
        assert rows[0][2] == 1, f"Expected shop_id=1, got {rows[0][2]}"

        # Check daily_metrics
        cur.execute("SELECT date, shop_id FROM daily_metrics WHERE CURRENT_DATE = date")
        rows = cur.fetchall()
        print(f"daily_metrics rows: {rows}")
        assert len(rows) > 0, "No row found in daily_metrics"
        assert rows[0][1] == 1, f"Expected shop_id=1, got {rows[0][1]}"

        # Check customer_demographics
        cur.execute("SELECT id, track_id, shop_id FROM customer_demographics WHERE track_id = %s", (test_track_id,))
        rows = cur.fetchall()
        print(f"customer_demographics rows: {rows}")
        assert len(rows) > 0, "No row found in customer_demographics"
        assert rows[0][2] == 1, f"Expected shop_id=1, got {rows[0][2]}"

        print("ALL 5 TABLES VERIFIED WITH shop_id = 1!")

    finally:
        # Cleanup test records
        cur.execute("DELETE FROM customer_entries WHERE track_id = %s", (test_track_id,))
        cur.execute("DELETE FROM customer_exits WHERE track_id = %s", (test_track_id,))
        cur.execute("DELETE FROM store_occupancy WHERE camera_id = %s", (test_cam,))
        cur.execute("DELETE FROM customer_demographics WHERE track_id = %s", (test_track_id,))
        cur.close()
        conn.close()
        print("Cleaned up test records successfully.")

if __name__ == "__main__":
    test_db_manager_shop_id()
