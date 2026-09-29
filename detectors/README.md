# Person-Bag Associator

`BagPersonAassociator` matches tracked people with tracked bags. It uses bounding-box overlap, distance, and the bag's position relative to the person. A candidate must remain valid for several consecutive frames before it becomes a confirmed association.

The implementation is in [`person_bag_associator.py`](./person_bag_associator.py). The class name contains the existing spelling `Aassociator`; use that name when importing it:

```python
from detectors.person_bag_associator import BagPersonAassociator

associator = BagPersonAassociator()
```

## Detection input

### `associate()`

The regular association method expects each person detection to contain:

```python
{
    "track_id": 12,
    "bbox": (x1, y1, x2, y2),
    "person_lower_body_midpoint": (center_x, bottom_y),
}
```

### `associate_person_bag_exit()`

The exit-specific method expects the person detection to contain:

```python
{
    "track_id": 12,
    "bbox": (x1, y1, x2, y2),
    "center": (center_x, upper_y),
}
```

Both methods expect each bag detection to contain:

```python
{
    "track_id": 7,
    "bbox": (x1, y1, x2, y2),
    "bag_midpoint": (center_x, center_y),
}
```

All detection lists can be empty when no tracked object has a valid YOLO track ID. The required keys must still be present on every returned detection.

## Association process

For every person, the associator evaluates every unused bag:

1. **Intersection over Bag (IoB)** measures how much of the bag box overlaps the person box.
2. **Distance score** rewards a bag close to the person midpoint.
3. **Position score** rewards a bag that is horizontally near the person and located in the lower part of the person box.
4. The final score is calculated as:

   ```text
   0.4 * IoB score
   + 0.4 * distance score
   + 0.2 * position score
   ```

A candidate is accepted only when:

- the bag is within `MAX_BAG_DISTANCE`;
- its IoB reaches `MIN_IOB`, or its relative position is acceptable; and
- the final score reaches `ASSOCIATION_THRESHOLD`.

Only the highest-scoring valid bag is selected for each person.

## Temporal confirmation

A valid candidate is stored under the pair `(person_track_id, bag_track_id)`. Its consecutive-frame count increases when the same pair is found again. The pair is confirmed after `MIN_CONSECUTIVE_FRAMES` successful frames.

The defaults are:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `min_consecutive_frames` | `5` | Frames required to confirm a pair |
| `max_bag_distance` | `150` | Maximum person-bag midpoint distance |
| `min_iob` | `0.10` | Minimum overlap ratio used by scoring |
| `association_threshold` | `0.50` | Minimum final association score |

Configure these values when constructing the associator:

```python
associator = BagPersonAassociator(
    min_consecutive_frames=5,
    max_bag_distance=150,
    min_iob=0.10,
    association_threshold=0.50,
)
```

## Public methods

### `associate(detections, bag_detections)`

Performs regular person-bag association using `person_lower_body_midpoint`.

When a pair is confirmed, it stores the pair and increments `person_with_bag_count`. Its current result contains:

```python
{
    "person_with_bag": 1,
    "score": 0.82,
}
```

The current regular result does not include `person_track_id` or `bag_track_id`; use `get_associations()` to retrieve the stored mapping.

### `associate_person_bag_exit(detections, bag_detections)`

Performs the same matching and temporal confirmation process using the person's `center` field. It returns confirmed exit candidates in this form:

```python
{
    "person_with_bag": 1,
    "person_track_id": 12,
    "bag_track_id": 7,
    "score": 0.82,
}
```

This result can be passed to `PersonWithBagExitValidator.update_associations()`.

### `get_associations()`

Returns a copy of the confirmed mapping:

```python
{
    person_track_id: bag_track_id,
}
```

### `get_count()`

Returns the total number of confirmed person-bag associations.

### `reset()`

Clears candidate history, confirmed mappings, used IDs, and the total count.

### `draw_confirmed_associations(frame, detections, bag_detections)`

Draws confirmed person and bag boxes, midpoint markers, a connecting line, and the current association score onto the OpenCV frame. It returns the modified frame.

## Typical integration

For the exit-validation flow, use the person detector output containing `center`:

```python
person_detections = person_detector.detect(frame)
bag_detections = bag_detector.detect_bags(frame)

exit_associations = associator.associate_person_bag_exit(
    person_detections,
    bag_detections,
)

exit_validator.update_associations(
    exit_associations,
    person_detections,
)
exit_events = exit_validator.update(person_detections)
```

For regular association, use the detector output containing `person_lower_body_midpoint`:

```python
person_detections = person_detector.detect_persons(frame)
bag_detections = bag_detector.detect_bags(frame)

associator.associate(person_detections, bag_detections)
```

## Important implementation notes

- `associate()` and `associate_person_bag_exit()` share candidate history, used-person IDs, used-bag IDs, confirmed mappings, and the total counter. Running both on the same associator can cause one method to suppress results from the other.
- Track IDs must be stable across frames. If a tracker loses and recreates an ID, temporal confirmation starts again.
- The matching methods use different person midpoint keys. A unified detector output containing both midpoint fields avoids schema errors when multiple consumers use the same detections.
- The drawing method currently reads `person_lower_body_midpoint`, so detections passed to it must include that field.
- A candidate that is not valid in a frame does not explicitly remove every prior candidate pair for that person; stale candidate history can remain until it is replaced or removed after confirmation.
