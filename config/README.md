# Configuration Settings

This directory contains the runtime constants used by the MCP Tracking application. Edit [`settings.py`](settings.py) to change model, detection, tracking, gate, and video behavior.

## Settings

| Variable | Current value | Purpose | Used by |
| --- | --- | --- | --- |
| `MODEL_PATH` | `models/yolov8m.pt` | Path to the YOLO weights loaded for person detection and tracking. | `detectors/person_detector.py` (`YOLO(...)`) |
| `TRACKER_CONFIG` | `botsort.yaml` | Ultralytics tracker configuration passed to `model.track()`. | `detectors/person_detector.py` |
| `PERSON_CLASS_ID` | `0` | COCO class ID for `person`; limits detection results to people. | `detectors/person_detector.py` (`classes=[...]`) |
| `CONFIDENCE` | `0.40` | Minimum confidence threshold for detections passed to the tracker. | `detectors/person_detector.py` (`conf=...`) |
| `IMAGE_SIZE` | `640` | Inference image size supplied to Ultralytics. | `detectors/person_detector.py` (`imgsz=...`) |
| `FRAME_PROCESS_INTERVAL` | `5` | Intended interval for processing frames, likely to reduce compute load. | Currently unused; no module imports it. |
| `TRAIL_LENGTH` | `30` | Maximum number of recent center points retained for each tracked person. | `tracking/track_manager.py` (`deque(maxlen=...)`) |
| `GATE_BAND_PIXELS` | `15` | Width, in pixels, of the neutral gate band. A point within this distance of the gate line is classified as `GATE`. | `gate/geometry.py` (`get_side()`) |
| `EVENT_COOLDOWN_SECONDS` | `2.0` | Intended cooldown between repeated entry or exit events. | Currently unused; event timing is not read from this setting. |
| `GATE_CONFIG_FILE` | `config/gate_config.json` | JSON file used to load and save gate endpoints and the inside reference point. | `gate/gate_manager.py` |
| `VIDEO_SOURCE` | `D:\\MCP_Tracking\\Videos\\Outlet_Sample_2_Banasree.avi` | Default video file or camera source for the application. Use an integer such as `0` for a camera. | Imported by `app.py`, but currently overridden by a local `VIDEO_SOURCE` assignment before `CustomerCounter` is created. |

## How Settings Flow Through the Application

1. `app.py` starts the application and passes a video source to `CustomerCounter`.
2. `services/customer_counter.py` creates the detector, camera, tracker, gate manager, and event manager.
3. `detectors/person_detector.py` uses the model and detection settings to produce tracked person detections.
4. `tracking/track_manager.py` stores each person's recent center points using `TRAIL_LENGTH`.
5. `gate/gate_manager.py` loads and saves the interactive gate configuration using `GATE_CONFIG_FILE`.
6. `gate/geometry.py` uses `GATE_BAND_PIXELS` to classify a tracked point as `INSIDE`, `OUTSIDE`, or `GATE`.

## Editing Guidance

- Relative paths such as `models/yolov8m.pt` and `config/gate_config.json` are resolved from the process working directory. Run the application from the project root so these paths resolve correctly.
- Increase `CONFIDENCE` to reduce false detections; decrease it to detect people more aggressively.
- Increase `IMAGE_SIZE` when small or distant people are missed, at the cost of performance.
- Increase `TRAIL_LENGTH` for longer visible tracks, at the cost of memory and drawing work.
- Increase `GATE_BAND_PIXELS` when detections jitter around the gate line and produce unstable side classifications.
- To make `VIDEO_SOURCE` configurable, remove the local assignment in `app.py` and pass the imported setting directly to `CustomerCounter`.
- `FRAME_PROCESS_INTERVAL` and `EVENT_COOLDOWN_SECONDS` should not be expected to change behavior until they are connected to the processing loop and event manager respectively.
