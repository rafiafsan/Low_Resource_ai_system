from datetime import datetime
from typing import Optional

from gate.geometry import get_side

MIN_ENTRY_SECONDS = 60
MIN_EXIT_SECONDS = 30


class EventManager:
    def __init__(
        self,
        min_entry_dwell_seconds: float = 60,
        min_exit_dwell_seconds: float = 10,
        deadband_pixels: int = 15,
        min_dwell_seconds: Optional[float] = None,
    ):
        if min_dwell_seconds is not None:
            self.min_entry_dwell_seconds = float(min_dwell_seconds)
            self.min_exit_dwell_seconds = float(min_dwell_seconds)
        else:
            self.min_entry_dwell_seconds = float(min_entry_dwell_seconds)
            self.min_exit_dwell_seconds = float(min_exit_dwell_seconds)

        self.min_dwell_seconds = self.min_entry_dwell_seconds
        self.deadband_pixels = deadband_pixels

        self.entry_count = 0
        self.exit_count = 0
        self.previous_side = {}
        self.pending_entries = {}
        self.pending_exits = {}
        self.completed_tracks = set()

        self.validated_entry_count = 0
        self.validated_exit_count = 0

    def evict_lost_tracks(self, lost_ids):
        """Evict lost tracks from previous_side to prevent unbounded memory growth."""
        if not lost_ids:
            return
        for tid in lost_ids:
            self.previous_side.pop(tid, None)

    def process(self, track_id, center, gate_points, inside_reference):
        if not gate_points or len(gate_points) < 2 or not inside_reference:
            return None
        
        if track_id in self.completed_tracks:
            return None

        current_side = get_side(center, gate_points[0], gate_points[1], inside_reference)
        
        if current_side == "GATE":
            return None

        previous = self.previous_side.get(track_id)
        self.previous_side[track_id] = current_side

        if previous is None:
            return None
        
        current_time = datetime.now()

        # ENTRY

        if previous == "OUTSIDE" and current_side == "INSIDE":

            self.entry_count += 1

            # Cancel pending EXIT
            if track_id in self.pending_exits:

                del self.pending_exits[track_id]

                self.exit_count -= 1

                return {
                    "track_id": track_id,
                    "event_type": "ENTRY",
                    "timestamp": current_time,
                    "validated": False,
                    "cancelled_exit": True
                }

            self.pending_entries[track_id] = current_time

            return {
                "track_id": track_id,
                "event_type": "ENTRY",
                "timestamp": current_time,
                "validated": False
            }


    
        # EXIT

        if previous == "INSIDE" and current_side == "OUTSIDE":

            self.exit_count += 1

            # Cancel pending ENTRY
            if track_id in self.pending_entries:

                del self.pending_entries[track_id]

                self.entry_count -= 1

                self.pending_exits[track_id] = current_time

                return {
                    "track_id": track_id,
                    "event_type": "EXIT",
                    "timestamp": current_time,
                    "validated": False,
                    "cancelled_entry": True
                }

            self.pending_exits[track_id] = current_time

            return {
                "track_id": track_id,
                "event_type": "EXIT",
                "timestamp": current_time,
                "validated": False
            }

        return None


    def validate_pending(self):
        current_time = datetime.now()

        validated_events = []


        # VALIDATE ENTRY
        
        for track_id in list(self.pending_entries.keys()):

            entry_time = self.pending_entries[track_id]

            elapsed_seconds = (
                current_time - entry_time
            ).total_seconds()

            if elapsed_seconds >= self.min_entry_dwell_seconds:

                self.validated_entry_count += 1

                del self.pending_entries[track_id]

                self.completed_tracks.add(track_id)

                validated_events.append({
                    "track_id": track_id,
                    "event_type": "ENTRY",
                    "timestamp": entry_time,
                    "validated": True,
                    "duration_seconds": elapsed_seconds
                })


        # VALIDATE EXIT

        for track_id in list(self.pending_exits.keys()):

            exit_time = self.pending_exits[track_id]

            elapsed_seconds = (
                current_time - exit_time
            ).total_seconds()

            if elapsed_seconds >= self.min_exit_dwell_seconds:

                self.validated_exit_count += 1

                del self.pending_exits[track_id]

                self.completed_tracks.add(track_id)

                validated_events.append({
                    "track_id": track_id,
                    "event_type": "EXIT",
                    "timestamp": exit_time,
                    "validated": True,
                    "duration_seconds": elapsed_seconds
                })

        return validated_events