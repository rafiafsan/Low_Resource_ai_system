# Retail AI Analytics Engine (Edge Low-Resource System)

A real-time, low-resource computer vision analytics system engineered for edge deployments (Intel Mini PC, 4 CPU cores, ~2 GB RAM, No GPU).

---

## Key Features

* **Footfall Analytics**: Accurate bidirectional customer Entry and Exit counting with dual-gate validation.
* **Dwell Time Validation**: Anti-bounce filters (Entry dwell: **60s**, Exit dwell: **10s**) to eliminate false triggers from loitering staff or gate wanderers.
* **Demographic Profiling**: MiVOLO age & gender classification (optimized direct person inference, lazy-loaded on CPU).
* **Live Occupancy Snapshotting**: Periodic calculation of active store visitors sent to the central database every 20 minutes.
* **Decoupled API Architecture**: Detection runs locally; events are queued asynchronously in memory and sent via HTTP POST to the central FastAPI gateway server (no direct PostgreSQL port exposure needed).
* **Zero-Drop Tracking**: Multi-threaded architecture keeps the RTSP grabber, YOLO detector, ByteTrack tracker, and HTTP sender decoupled.

---

## System Architecture

```
     Retail Outlet (Edge Mini PC)                      Company Central Server
  ┌───────────────────────────────┐               ┌───────────────────────────────┐
  │         CCTV Camera           │               │                               │
  │     (RTSP / Video File)       │               │                               │
  └───────────────┬───────────────┘               │                               │
                  │                               │                               │
                  ▼                               │                               │
  ┌───────────────────────────────┐               │                               │
  │     RetailAI_v2.exe / app.py  │               │                               │
  │  - YOLOv8 ONNX (320x320)      │   HTTP POST   │     FastAPI Gateway Server    │
  │  - ByteTrack (Person Tracking)├──────────────►│      (Port 8000 / HTTPS)      │
  │  - MiVOLO Demographics        │  (Non-blocking│                               │
  │  - Non-blocking Queue         │   Queue)      └───────────────┬───────────────┘
  └───────────────────────────────┘                               │
                                                                  ▼
                                                  ┌───────────────────────────────┐
                                                  │      PostgreSQL Database      │
                                                  │  - customer_entries           │
                                                  │  - customer_exits             │
                                                  │  - customer_demographics      │
                                                  │  - store_occupancy            │
                                                  │  - daily_metrics              │
                                                  └───────────────────────────────┘
```

---

## Prerequisites & Requirements

* **OS**: Windows 10 / 11 (64-bit) or Linux
* **Python**: **Python 3.12** is required (System packages are installed in Python 3.12).
* **Target Hardware**: Intel Celeron / Pentium / Core i3/i5 (4 Cores, 4 GB RAM, 2 GB usable).

> [!IMPORTANT]
> Always use `py -3.12` to run scripts instead of generic `python` (which may default to Python 3.14 without dependencies installed).

---

## How to Run the Project

### Option 1: Running with Python 3.12 (Development Mode)

From the project root directory (`D:\Low_Resource_ai_system`):

```powershell
# 1. Run with default configuration (uses fallback test video)
py -3.12 app.py

# 2. Run with a specific local video file
py -3.12 app.py --source "test_videos/Outlet_Sample_1_Banasree.avi"

# 3. Run with a live RTSP CCTV camera stream
py -3.12 app.py --source "rtsp://admin:password123@192.168.1.100:554/h264Preview_01_main"
```

---

### Option 2: Running Standalone Executable (Production / Other PC)

The project includes a fully pre-compiled standalone package that requires **no Python installation, no virtual environment, and no external dependencies**:

1. Navigate to:
   ```
   D:\Low_Resource_ai_system\dist\RetailAI_v2\
   ```
2. Double-click **`run_retail_ai.bat`** (or launch `RetailAI_v2.exe`).

To run the EXE on a specific video or camera:
```cmd
cd /d D:\Low_Resource_ai_system\dist\RetailAI_v2
RetailAI_v2.exe --source "path\to\video.avi"
```

---

### Option 3: Starting the Central FastAPI Gateway Server

If running the gateway server that receives data from the outlets:

```powershell
# 1. Install gateway dependencies (on the server)
pip install fastapi uvicorn psycopg2-binary pydantic

# 2. Start the FastAPI server (Port 8000)
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Configuration Guide

All system settings are organized in [`config/production.yaml`](file:///d:/Low_Resource_ai_system/config/production.yaml):

| Setting | File / Key | Default | Description |
| :--- | :--- | :--- | :--- |
| **Video Source** | `camera.fallback_video` | `test_videos/Outlet_Sample_2_Banasree.avi` | Default test video when no `--source` argument is passed. |
| **Inference Model** | `inference.model` | `models/yolov8m.onnx` | Primary detection model (ONNX Runtime CPU). |
| **AI Target FPS** | `inference.target_fps` | `4` | Target inference frame rate (limits CPU usage). |
| **Entry Validation** | `gates.min_entry_dwell_seconds` | `60` | Customer must remain inside for 60s before entry is validated. |
| **Exit Validation** | `gates.min_exit_dwell_seconds` | `10` | Exit confirmation threshold (seconds). |
| **Female Threshold** | `demographics.female_threshold`| `0.50` | Confidence threshold for female classification. |
| **Occupancy Interval**| `database.occupancy_interval_minutes`| `20.0` | Frequency (minutes) of store occupancy snapshots. |

### Configuring API Server Connection for Outlets

In [`database/db_manager.py`](file:///d:/Low_Resource_ai_system/database/db_manager.py) or via environment variables:

```ini
API_GATEWAY_URL=http://172.17.2.227:8001
API_KEY=secret_outlet_token_banasree
SHOP_ID=2
```

On another outlet PC, you can configure these directly inside `run_retail_ai.bat`:
```bat
@echo off
set API_GATEWAY_URL=http://172.17.2.227:8001
set API_KEY=secret_outlet_token_banasree
set SHOP_ID=2
RetailAI_v2.exe
```

---

## Building a New Executable

Whenever you make source code modifications and want to compile a new `.exe`:

```powershell
py -3.12 build_exe_v2.py
```

This compiles the PyInstaller bundle and copies all runtime assets (`config/`, `models/`, `mivolo_system/model/`, and `test_videos/`) into `dist/RetailAI_v2/`.

---

## Keyboard Controls (During Live Preview)

| Key | Action |
| :---: | :--- |
| **`q`** | Safely quit the application, flush all pending API events, and release camera resources. |
| **`p`** | Pause / Resume video playback. |
| **`h`** | Toggle on-screen Performance HUD overlay. |

---

## Project Structure

```
Low_Resource_ai_system/
├── app.py                      # Main application entry point
├── build_exe_v2.py             # Standalone PyInstaller build script
├── RetailAI_v2.spec            # PyInstaller specification file
├── camera/
│   └── rtsp_reader.py          # Decoupled low-latency RTSP / video reader
├── config/
│   ├── production.yaml         # Master production configuration
│   ├── gates.json              # Interactive gate coordinates
│   └── database.json           # Central database credentials
├── database/
│   └── db_manager.py           # Non-blocking HTTP API Gateway manager
├── detectors/
│   ├── person_bag_associator.py# Links customers to carried bags
│   └── person_bag_exit_validator.py
├── gate/
│   ├── event_manager.py        # Validated Entry / Exit dwell logic
│   ├── geometry.py             # Vector cross-product gate side calculation
│   └── unified_gate_manager.py # Gate coordinates loader
├── inference/
│   ├── detector.py             # Detector factory
│   └── onnx_detector.py        # ONNX Runtime CPU multi-threaded detector
├── mivolo_system/
│   └── async_worker.py         # Asynchronous MiVOLO age & gender profiler
├── models/
│   ├── yolov8m.onnx            # 320x320 lightweight ONNX model
│   └── yolov8m.pt              # Full PyTorch model weights
├── services/
│   ├── business_logic.py       # Hook interfaces for entry/exit events
│   └── unified_pipeline.py     # Central analytics pipeline coordinator
├── test_videos/                # Sample video footage for calibration
├── tracking/
│   └── track_manager.py        # ByteTrack multi-object tracker
└── dist/
    └── RetailAI_v2/            # Packaged standalone distribution for Mini PC
```
