"""Business Logic Hook Interfaces
=================================
Provides modular, customizable hooks for retail analytics events.
The core infrastructure (RTSP reader, inference scheduler, database persistence)
calls these clean hook functions, allowing business logic to be swapped or extended
without touching the underlying hardware-optimized streaming pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple


def process_entry_event(
    track_id: int,
    timestamp: str,
    event_dict: Dict[str, Any],
    demographics: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Hook invoked when a validated customer entry transition occurs.

    Args:
        track_id: Persistent tracking ID.
        timestamp: String formatted timestamp of entry.
        event_dict: Event details from EventManager.
        demographics: Estimated demographics if available.

    Returns:
        Structured record to persist, or None to discard.
    """
    age_group = None
    gender = None

    if demographics:
        age_group = demographics.get("age_group")
        gender = demographics.get("gender")

    return {
        "track_id": track_id,
        "timestamp": timestamp,
        "direction": "ENTRY",
        "age_group": age_group,
        "gender": gender,
    }


def process_exit_event(
    track_id: int,
    timestamp: str,
    event_dict: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Hook invoked when a validated customer exit transition occurs.

    Args:
        track_id: Persistent tracking ID.
        timestamp: String formatted timestamp of exit.
        event_dict: Event details from EventManager.

    Returns:
        Structured record to persist, or None to discard.
    """
    return {
        "track_id": track_id,
        "timestamp": timestamp,
        "direction": "EXIT",
    }


def validate_person_bag_exit(
    person_track_id: int,
    bag_track_id: int,
    event_dict: Dict[str, Any],
) -> bool:
    """Validation hook for confirming a person left with a shopping bag.

    Args:
        person_track_id: Tracking ID of person.
        bag_track_id: Tracking ID of associated bag.
        event_dict: Exit details.

    Returns:
        True if event is a confirmed person-with-bag exit, False otherwise.
    """
    return event_dict.get("event") == "EXIT_WITH_BAG"


def validate_customer_event(event_type: str, event_data: Dict[str, Any]) -> bool:
    """General validation hook for arbitrary retail events."""
    if event_type in ("ENTRY", "EXIT"):
        return not event_data.get("cancelled_exit") and not event_data.get("cancelled_entry")
    return True
