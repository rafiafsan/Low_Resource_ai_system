# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

block_cipher = None

PROJECT_ROOT = os.path.abspath(".")
MIVOLO_ROOT = os.path.join(PROJECT_ROOT, "mivolo_system")

hidden_imports = [
    # Core Frameworks
    "torch",
    "torchvision",
    "onnxruntime",
    "cv2",
    "numpy",
    "psutil",
    "yaml",
    "requests",
    "supervision",
    "omegaconf",
    "antlr4",
    
    # Ultralytics Dynamic Modules
    "ultralytics",
    "ultralytics.nn",
    "ultralytics.nn.modules",
    "ultralytics.nn.modules.conv",
    "ultralytics.nn.modules.block",
    "ultralytics.nn.modules.head",
    "ultralytics.models",
    "ultralytics.models.yolo",
    "ultralytics.models.yolo.detect",
    "ultralytics.engine",
    "ultralytics.engine.results",
    "ultralytics.utils",
    
    # Scipy & Timm
    "scipy",
    "scipy.special",
    "scipy.spatial",
    "timm",
    "timm.models",
    
    # Internal Modules
    "camera.rtsp_reader",
    "config.settings",
    "database.db_manager",
    "detectors.person_bag_associator",
    "detectors.person_bag_exit_validator",
    "gate.event_manager",
    "gate.geometry",
    "gate.unified_gate_manager",
    "inference.detector",
    "inference.onnx_detector",
    "inference.torch_detector",
    "inference.inference_scheduler",
    "mivolo_system.async_worker",
    "monitoring.performance_monitor",
    "services.business_logic",
    "services.unified_pipeline",
    "tracking.track_manager",
    "visualization.draw_track",
    "visualization.dashboard",
    
    # MiVOLO Components
    "MiVOLO",
    "MiVOLO.predictor",
    "MiVOLO.structures",
    "MiVOLO.data",
    "MiVOLO.data.misc",
    "model.mi_volo",
    "model.yolo_detector",
    "model.create_timm_model",
    "model.mivolo_model",
    "model.cross_bottleneck_attn",
]

datas = [
    ("config", "config"),
]

binaries = []

excludes = [
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
    "playwright",
    "matplotlib.tests",
    "tkinter.test",
]

a = Analysis(
    ["app.py"],
    pathex=[PROJECT_ROOT, MIVOLO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RetailAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="RetailAI",
)
