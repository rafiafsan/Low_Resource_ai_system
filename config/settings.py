import os

# =============================================================================
# Model & Tracking Weights
# =============================================================================
# Legacy default model (COCO 80 classes)
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8l.pt")

# Primary Model: Person & Bag detection with BoT-SORT tracking
PRIMARY_MODEL_PATH = os.getenv("PRIMARY_MODEL_PATH", "models/yolov8l.pt")

# Secondary Detector: Fast Bag detection (COCO classes 24, 26)
BAG_MODEL_PATH = os.getenv("BAG_MODEL_PATH", "models/yolov8m.pt")

# MiVOLO Demographic Age/Gender Checkpoint
MIVOLO_CHECKPOINT = os.getenv(
    "MIVOLO_CHECKPOINT", "mivolo_system/model/model_imdb_cross_person_4.22_99.46.pth.tar"
)

TRACKER_CONFIG = "botsort.yaml"

PERSON_CLASS_ID = 0
BAG_CLASS_ID = 26

CONFIDENCE = float(os.getenv("CONFIDENCE", "0.40"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

TRAIL_LENGTH = int(os.getenv("TRAIL_LENGTH", "30"))
GATE_BAND_PIXELS = int(os.getenv("GATE_BAND_PIXELS", "15"))

# =============================================================================
# Gate Configurations
# =============================================================================
# Unified multi-detector gate file
GATES_CONFIG_FILE = "config/gates.json"

# Legacy single-module gate files (for backward compatibility)
GATE_CONFIG_FILE = "config/gate_config.json"
GATE_CONFIG_FILE_PERSON_BAG = "config/gate_config_bag.json"
GATE_CONFIG_FILE_AGE_GENDER = "config/age_gender_gate_config.json"

# =============================================================================
# Default Video Source (RTSP stream URL, local file, or camera index)
# =============================================================================
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", r"D:\MCP_Tracking\Videos\Outlet_Sample_3_Banasree.avi")