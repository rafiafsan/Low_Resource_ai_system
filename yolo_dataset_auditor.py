"""
YOLO Dataset Auditor & Visualizer
==================================
A production-ready desktop application built with Python, Tkinter, and Pillow
to view, verify, and audit object detection datasets formatted in standard YOLO format.

Key Features:
- Supports: .jpg, .jpeg, .png, .bmp, .webp
- Normalized YOLO bounding box parsing (<class_id> <x_center> <y_center> <width> <height> [confidence])
- Responsive Canvas with aspect-ratio preservation and debounced redraws
- Interactive Zoom (Mouse Wheel / +/-) and Pan (Click & Drag) with cursor centering
- Auto-detection of sibling /labels folder when selecting an /images folder
- Auto-loading of class definitions from dataset.yaml, data.yaml, or classes.txt
- Dynamic per-class visibility checkboxes and color-coded legend
- Audit actions:
    * 'F' to flag/unflag current image (appends to flagged_for_review.txt)
    * 'Delete' to move current label file to a trash/ directory (with Ctrl+Z undo support)
- Keyboard shortcuts: 'A'/'D' or Left/Right arrows for navigation, 'R' to reset zoom, etc.
- Modern dark-mode UI with high-contrast bounding box tags and status indicators
"""

import os
import re
import sys
import glob
import shutil
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps


# =============================================================================
# High-DPI Display Support on Windows
# =============================================================================
try:
    import ctypes
    # Per-monitor DPI awareness (Windows 8.1+)
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        # Fallback for older Windows
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# =============================================================================
# Color Palette & Theme Definitions
# =============================================================================
THEME = {
    "bg_dark": "#121215",
    "bg_panel": "#1b1b20",
    "bg_surface": "#24242c",
    "bg_surface_hover": "#32323e",
    "border": "#363644",
    "canvas_bg": "#09090b",
    "text_primary": "#f4f4f6",
    "text_secondary": "#a1a1aa",
    "text_muted": "#71717a",
    "accent_blue": "#3b82f6",
    "accent_blue_hover": "#2563eb",
    "accent_green": "#10b981",
    "accent_orange": "#f59e0b",
    "accent_red": "#ef4444",
    "flag_badge": "#ff9800",
}

# Pre-defined default classes
DEFAULT_CLASSES: Dict[int, Dict[str, str]] = {
    0: {"name": "person", "color": "#00E5FF"},   # Cyan
    1: {"name": "face", "color": "#76FF03"},     # Green
    2: {"name": "handbag", "color": "#FF1744"},  # Red
}

# Extended color palette for additional classes loaded dynamically
EXTENDED_PALETTE: List[str] = [
    "#FFD600",  # Vivid Yellow
    "#D500F9",  # Vivid Magenta / Purple
    "#FF6D00",  # Deep Orange
    "#2979FF",  # Royal Blue
    "#00E676",  # Bright Mint
    "#FF4081",  # Neon Pink
    "#00B0FF",  # Light Blue
    "#7C4DFF",  # Deep Purple
    "#FFAB00",  # Amber
    "#1DE9B6",  # Teal
    "#C6FF00",  # Lime
    "#FF3D00",  # Red-Orange
    "#E040FB",  # Orchid
    "#00E5FF",  # Cyan
    "#76FF03",  # Light Green
]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# Data Structures
# =============================================================================
class YoloBBox:
    """Represents a single normalized YOLO bounding box."""
    def __init__(self, class_id: int, xc: float, yc: float, w: float, h: float, conf: Optional[float] = None):
        self.class_id = int(class_id)
        # Clamp normalized coordinates strictly within [0.0, 1.0]
        self.xc = max(0.0, min(1.0, float(xc)))
        self.yc = max(0.0, min(1.0, float(yc)))
        self.w = max(0.0, min(1.0, float(w)))
        self.h = max(0.0, min(1.0, float(h)))
        self.conf = float(conf) if conf is not None else None

    def to_original_pixels(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert normalized [xc, yc, w, h] to absolute pixel coordinates [x1, y1, x2, y2]."""
        x1 = (self.xc - self.w / 2.0) * img_w
        y1 = (self.yc - self.h / 2.0) * img_h
        x2 = (self.xc + self.w / 2.0) * img_w
        y2 = (self.yc + self.h / 2.0) * img_h
        return x1, y1, x2, y2


# =============================================================================
# Main Application Class
# =============================================================================
class YoloDatasetAuditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Dataset Auditor & Visualizer")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 650)
        self.root.configure(bg=THEME["bg_dark"])

        # Dataset State
        self.images_dir: Optional[Path] = None
        self.labels_dir: Optional[Path] = None
        self.image_files: List[Path] = []
        self.current_idx: int = -1
        self.current_pil_image: Optional[Image.Image] = None
        self.current_bboxes: List[YoloBBox] = []

        # Audit State
        self.flagged_files: Set[str] = set()
        self.flagged_log_path: Optional[Path] = None
        self.last_trashed_info: Optional[Tuple[Path, Path]] = None  # (trash_path, original_path)

        # Classes & Colors
        self.classes: Dict[int, Dict[str, str]] = dict(DEFAULT_CLASSES)
        self.class_visibility: Dict[int, tk.BooleanVar] = {}

        # Canvas & View State
        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.is_dragging: bool = False
        self.drag_start_x: int = 0
        self.drag_start_y: int = 0
        self.cached_tk_image: Optional[ImageTk.PhotoImage] = None
        self.show_labels_text: bool = True
        self.show_bboxes: bool = True
        self._resize_job: Optional[str] = None

        # Build UI
        self._init_ttk_styles()
        self._build_layout()
        self._bind_shortcuts()

        # Check for immediate default dataset in current working directory
        self._auto_load_default_dataset()

    # -------------------------------------------------------------------------
    # Styling & Layout
    # -------------------------------------------------------------------------
    def _init_ttk_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=THEME["bg_panel"], foreground=THEME["text_primary"], font=("Segoe UI", 9))
        style.configure("TFrame", background=THEME["bg_panel"])
        style.configure("TLabel", background=THEME["bg_panel"], foreground=THEME["text_primary"])
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=THEME["text_primary"])
        style.configure("Muted.TLabel", font=("Segoe UI", 9), foreground=THEME["text_secondary"])

        # Custom Buttons
        style.configure(
            "Dark.TButton",
            background=THEME["bg_surface"],
            foreground=THEME["text_primary"],
            borderwidth=1,
            focuscolor=THEME["accent_blue"],
            relief="flat",
            padding=(10, 6)
        )
        style.map(
            "Dark.TButton",
            background=[("active", THEME["bg_surface_hover"]), ("pressed", THEME["border"])],
            foreground=[("active", "#ffffff")]
        )

        style.configure(
            "Primary.TButton",
            background=THEME["accent_blue"],
            foreground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
            relief="flat",
            padding=(12, 6)
        )
        style.map(
            "Primary.TButton",
            background=[("active", THEME["accent_blue_hover"]), ("pressed", "#1d4ed8")]
        )

        style.configure(
            "Danger.TButton",
            background=THEME["accent_red"],
            foreground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
            relief="flat",
            padding=(10, 5)
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#dc2626"), ("pressed", "#b91c1c")]
        )

        # Checkbutton
        style.configure(
            "Dark.TCheckbutton",
            background=THEME["bg_panel"],
            foreground=THEME["text_primary"],
            focuscolor=THEME["bg_panel"],
            font=("Segoe UI", 9)
        )
        style.map(
            "Dark.TCheckbutton",
            background=[("active", THEME["bg_panel"])],
            foreground=[("active", "#ffffff")]
        )

    def _build_layout(self):
        # 1. Top Control Bar
        self.top_bar = tk.Frame(self.root, bg=THEME["bg_panel"], height=52, padx=14, pady=8)
        self.top_bar.pack(side=tk.TOP, fill=tk.X)
        self._build_top_bar()

        # 2. Bottom Status Bar
        self.bottom_bar = tk.Frame(self.root, bg=THEME["bg_panel"], height=44, padx=14, pady=6)
        self.bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._build_bottom_bar()

        # 3. Main Center Area (Canvas + Sidebar)
        self.center_frame = tk.Frame(self.root, bg=THEME["bg_dark"])
        self.center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas Area (Left / Center)
        self.canvas_container = tk.Frame(self.center_frame, bg=THEME["canvas_bg"])
        self.canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.canvas_container,
            bg=THEME["canvas_bg"],
            highlightthickness=0,
            cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<Double-Button-1>", lambda e: self.reset_zoom())

        # Notification Banner (floating inside canvas)
        self.toast_label = tk.Label(
            self.canvas_container,
            text="",
            bg=THEME["bg_surface"],
            fg=THEME["text_primary"],
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=6,
            relief="solid",
            bd=1
        )
        self.toast_job = None

        # Sidebar (Right)
        self.sidebar = tk.Frame(self.center_frame, bg=THEME["bg_panel"], width=310, padx=14, pady=12)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

    def _build_top_bar(self):
        # Brand / Title
        title_lbl = tk.Label(
            self.top_bar,
            text="🎯 YOLO Auditor",
            font=("Segoe UI", 12, "bold"),
            bg=THEME["bg_panel"],
            fg=THEME["text_primary"]
        )
        title_lbl.pack(side=tk.LEFT, padx=(0, 15))

        # Folder Selection Buttons
        btn_img = ttk.Button(self.top_bar, text="📁 Select Images", style="Primary.TButton", command=self.select_images_folder)
        btn_img.pack(side=tk.LEFT, padx=(0, 8))

        btn_lbl = ttk.Button(self.top_bar, text="🏷️ Select Labels", style="Dark.TButton", command=self.select_labels_folder)
        btn_lbl.pack(side=tk.LEFT, padx=(0, 10))

        # Directory display label
        self.dir_status_lbl = tk.Label(
            self.top_bar,
            text="No dataset loaded. Click 'Select Images' to begin.",
            font=("Segoe UI", 9),
            bg=THEME["bg_panel"],
            fg=THEME["text_secondary"],
            anchor="w"
        )
        self.dir_status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Right side actions on Top Bar
        btn_load_yaml = ttk.Button(self.top_bar, text="⚙️ Load YAML", style="Dark.TButton", command=self.select_yaml_file)
        btn_load_yaml.pack(side=tk.RIGHT, padx=(6, 0))

        btn_reset_zoom = ttk.Button(self.top_bar, text="🔍 Reset Zoom (R)", style="Dark.TButton", command=self.reset_zoom)
        btn_reset_zoom.pack(side=tk.RIGHT, padx=(6, 0))

        btn_help = ttk.Button(self.top_bar, text="❓ Shortcuts", style="Dark.TButton", command=self.show_shortcuts_dialog)
        btn_help.pack(side=tk.RIGHT, padx=(6, 0))

    def _build_bottom_bar(self):
        # Prev / Next Controls
        self.btn_prev = ttk.Button(self.bottom_bar, text="◀ Prev (A)", style="Dark.TButton", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_next = ttk.Button(self.bottom_bar, text="Next (D) ▶", style="Dark.TButton", command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=(0, 14))

        # Direct Jump entry
        jump_lbl = tk.Label(self.bottom_bar, text="Go to:", bg=THEME["bg_panel"], fg=THEME["text_secondary"], font=("Segoe UI", 9))
        jump_lbl.pack(side=tk.LEFT, padx=(0, 4))

        self.jump_entry = tk.Entry(self.bottom_bar, width=5, bg=THEME["bg_surface"], fg=THEME["text_primary"], insertbackground="#fff", bd=1, relief="flat")
        self.jump_entry.pack(side=tk.LEFT, padx=(0, 4))
        self.jump_entry.bind("<Return>", self._on_jump_return)

        # Image Counter
        self.lbl_counter = tk.Label(
            self.bottom_bar,
            text="Image: 0 / 0",
            font=("Segoe UI", 9, "bold"),
            bg=THEME["bg_panel"],
            fg=THEME["text_primary"]
        )
        self.lbl_counter.pack(side=tk.LEFT, padx=(4, 16))

        # Flag Badge (Hidden by default)
        self.flag_badge = tk.Label(
            self.bottom_bar,
            text="⚠️ FLAGGED FOR REVIEW",
            font=("Segoe UI", 8, "bold"),
            bg=THEME["flag_badge"],
            fg="#000000",
            padx=8,
            pady=2
        )

        # File Name
        self.lbl_filename = tk.Label(
            self.bottom_bar,
            text="No file selected",
            font=("Segoe UI", 9),
            bg=THEME["bg_panel"],
            fg=THEME["text_primary"]
        )
        self.lbl_filename.pack(side=tk.LEFT, padx=8)

        # Right side info: Dimensions and Object count
        self.lbl_objects = tk.Label(
            self.bottom_bar,
            text="Objects: 0",
            font=("Segoe UI", 9, "bold"),
            bg=THEME["bg_panel"],
            fg=THEME["text_primary"]
        )
        self.lbl_objects.pack(side=tk.RIGHT, padx=10)

        self.lbl_dimensions = tk.Label(
            self.bottom_bar,
            text="-- × --",
            font=("Segoe UI", 9),
            bg=THEME["bg_panel"],
            fg=THEME["text_secondary"]
        )
        self.lbl_dimensions.pack(side=tk.RIGHT, padx=10)

    def _build_sidebar(self):
        # Sidebar Header
        sb_title = tk.Label(
            self.sidebar,
            text="Class Visibility",
            font=("Segoe UI", 11, "bold"),
            bg=THEME["bg_panel"],
            fg=THEME["text_primary"],
            anchor="w"
        )
        sb_title.pack(fill=tk.X, pady=(0, 6))

        # Toggle All / None
        toggle_bar = tk.Frame(self.sidebar, bg=THEME["bg_panel"])
        toggle_bar.pack(fill=tk.X, pady=(0, 8))

        btn_show_all = ttk.Button(toggle_bar, text="Show All", style="Dark.TButton", command=self.show_all_classes)
        btn_show_all.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        btn_hide_all = ttk.Button(toggle_bar, text="Hide All", style="Dark.TButton", command=self.hide_all_classes)
        btn_hide_all.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        # Scrollable Class List Container
        self.classes_scroll_frame = tk.Frame(self.sidebar, bg=THEME["bg_panel"])
        self.classes_scroll_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.classes_inner_frame = tk.Frame(self.classes_scroll_frame, bg=THEME["bg_panel"])
        self.classes_inner_frame.pack(fill=tk.BOTH, expand=True)

        self._refresh_class_legend_widgets()

        # Audit Actions Box
        sep = tk.Frame(self.sidebar, height=1, bg=THEME["border"])
        sep.pack(fill=tk.X, pady=8)

        audit_title = tk.Label(
            self.sidebar,
            text="Audit Controls",
            font=("Segoe UI", 10, "bold"),
            bg=THEME["bg_panel"],
            fg=THEME["text_primary"],
            anchor="w"
        )
        audit_title.pack(fill=tk.X, pady=(0, 6))

        self.btn_flag = ttk.Button(
            self.sidebar,
            text="🚩 Flag Current Image (F)",
            style="Dark.TButton",
            command=self.toggle_flag_current_image
        )
        self.btn_flag.pack(fill=tk.X, pady=(0, 6))

        self.btn_trash = ttk.Button(
            self.sidebar,
            text="🗑️ Move Label to Trash (Del)",
            style="Danger.TButton",
            command=self.trash_current_label
        )
        self.btn_trash.pack(fill=tk.X, pady=(0, 6))

        self.btn_undo_trash = ttk.Button(
            self.sidebar,
            text="↩️ Undo Trash (Ctrl+Z)",
            style="Dark.TButton",
            command=self.undo_trash_label
        )
        self.btn_undo_trash.pack(fill=tk.X, pady=(0, 6))

        # Open in Explorer
        btn_explore = ttk.Button(
            self.sidebar,
            text="📂 Open Dataset Folder",
            style="Dark.TButton",
            command=self.open_in_file_explorer
        )
        btn_explore.pack(fill=tk.X, pady=(6, 0))

    def _refresh_class_legend_widgets(self):
        """Re-generates the class checkboxes in the sidebar with live counts and color badges."""
        for widget in self.classes_inner_frame.winfo_children():
            widget.destroy()

        # Count occurrences in current image
        current_counts = {}
        for bbox in self.current_bboxes:
            current_counts[bbox.class_id] = current_counts.get(bbox.class_id, 0) + 1

        for cid, info in sorted(self.classes.items()):
            if cid not in self.class_visibility:
                self.class_visibility[cid] = tk.BooleanVar(value=True)

            row = tk.Frame(self.classes_inner_frame, bg=THEME["bg_panel"], pady=2)
            row.pack(fill=tk.X)

            # Color badge
            color_swatch = tk.Canvas(row, width=16, height=16, bg=THEME["bg_panel"], highlightthickness=0)
            color_swatch.pack(side=tk.LEFT, padx=(0, 6))
            color_swatch.create_oval(2, 2, 14, 14, fill=info["color"], outline="#ffffff", width=1)

            # Checkbutton
            chk = ttk.Checkbutton(
                row,
                text=f"{cid}: {info['name']}",
                variable=self.class_visibility[cid],
                style="Dark.TCheckbutton",
                command=self._on_class_visibility_toggled
            )
            chk.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Current count badge
            count = current_counts.get(cid, 0)
            cnt_lbl = tk.Label(
                row,
                text=f"x{count}",
                bg=THEME["bg_surface"] if count > 0 else THEME["bg_panel"],
                fg=THEME["text_primary"] if count > 0 else THEME["text_muted"],
                font=("Segoe UI", 8, "bold"),
                padx=5,
                pady=1
            )
            cnt_lbl.pack(side=tk.RIGHT)

    # -------------------------------------------------------------------------
    # Shortcuts & Key Bindings
    # -------------------------------------------------------------------------
    def _bind_shortcuts(self):
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<d>", lambda e: self.next_image())
        self.root.bind("<D>", lambda e: self.next_image())
        self.root.bind("<a>", lambda e: self.prev_image())
        self.root.bind("<A>", lambda e: self.prev_image())
        self.root.bind("<Home>", lambda e: self.go_to_first())
        self.root.bind("<End>", lambda e: self.go_to_last())

        # Audit Shortcuts
        self.root.bind("<f>", lambda e: self.toggle_flag_current_image())
        self.root.bind("<F>", lambda e: self.toggle_flag_current_image())
        self.root.bind("<Delete>", lambda e: self.trash_current_label())
        self.root.bind("<BackSpace>", lambda e: self.trash_current_label())
        self.root.bind("<Control-z>", lambda e: self.undo_trash_label())
        self.root.bind("<Control-Z>", lambda e: self.undo_trash_label())

        # View Shortcuts
        self.root.bind("<r>", lambda e: self.reset_zoom())
        self.root.bind("<R>", lambda e: self.reset_zoom())
        self.root.bind("<plus>", lambda e: self._zoom_by_factor(1.2))
        self.root.bind("<equal>", lambda e: self._zoom_by_factor(1.2))
        self.root.bind("<minus>", lambda e: self._zoom_by_factor(0.8))
        self.root.bind("<t>", lambda e: self.toggle_label_text())
        self.root.bind("<T>", lambda e: self.toggle_label_text())
        self.root.bind("<b>", lambda e: self.toggle_bboxes())
        self.root.bind("<B>", lambda e: self.toggle_bboxes())
        self.root.bind("<?>", lambda e: self.show_shortcuts_dialog())
        self.root.bind("<F1>", lambda e: self.show_shortcuts_dialog())

    # -------------------------------------------------------------------------
    # Dataset Loading & Auto-Detection
    # -------------------------------------------------------------------------
    def _auto_load_default_dataset(self):
        """Checks typical local paths like D:\\MCP_Tracking\\Annotated_Dataset\\images and loads them automatically."""
        candidates = [
            Path(r"D:\MCP_Tracking\Annotated_Dataset\images"),
            Path("./Annotated_Dataset/images"),
            Path("./Dataset"),
            Path(r"D:\MCP_Tracking\Dataset")
        ]
        for path in candidates:
            if path.exists() and path.is_dir():
                self._load_images_from_directory(path)
                return

    def select_images_folder(self):
        initial = str(self.images_dir) if self.images_dir else os.getcwd()
        selected = filedialog.askdirectory(title="Select Images Directory", initialdir=initial)
        if selected:
            self._load_images_from_directory(Path(selected))

    def select_labels_folder(self):
        initial = str(self.labels_dir) if self.labels_dir else os.getcwd()
        selected = filedialog.askdirectory(title="Select Labels Directory", initialdir=initial)
        if selected:
            self.labels_dir = Path(selected)
            self._update_dir_status()
            self._load_current_image()

    def select_yaml_file(self):
        initial = str(self.images_dir.parent) if self.images_dir else os.getcwd()
        selected = filedialog.askopenfilename(
            title="Select Dataset Configuration (YAML or TXT)",
            initialdir=initial,
            filetypes=[("YAML or Text Config", "*.yaml;*.yml;*.txt"), ("All Files", "*.*")]
        )
        if selected:
            self._parse_class_definitions(Path(selected))
            self._refresh_class_legend_widgets()
            self._render_canvas()
            self.show_toast("Loaded class definitions from file")

    def _load_images_from_directory(self, images_path: Path):
        self.images_dir = images_path

        # Auto-detect sibling /labels folder
        sibling_labels = images_path.parent / "labels"
        if sibling_labels.exists() and sibling_labels.is_dir():
            self.labels_dir = sibling_labels
            auto_detected = True
        elif (images_path / "labels").exists():
            self.labels_dir = images_path / "labels"
            auto_detected = True
        else:
            # Check if .txt files are directly alongside images
            txt_files = list(images_path.glob("*.txt"))
            if txt_files:
                self.labels_dir = images_path
                auto_detected = True
            else:
                self.labels_dir = sibling_labels  # fallback
                auto_detected = False

        # Scan for supported images
        self.image_files = sorted([
            f for f in images_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        # Auto-detect dataset.yaml / data.yaml / classes.txt
        yaml_candidates = [
            images_path.parent / "dataset.yaml",
            images_path.parent / "data.yaml",
            images_path / "dataset.yaml",
            images_path.parent / "classes.txt",
            images_path / "classes.txt"
        ]
        for ypath in yaml_candidates:
            if ypath.exists():
                self._parse_class_definitions(ypath)
                break

        # Setup flagged log path
        self.flagged_log_path = images_path.parent / "flagged_for_review.txt"
        self._load_flagged_list()

        self._update_dir_status(auto_detected)
        self.current_idx = 0 if self.image_files else -1
        self._load_current_image()

    def _update_dir_status(self, auto_detected: bool = False):
        if not self.images_dir:
            self.dir_status_lbl.config(text="No dataset loaded.")
            return

        lbl_text = f"🖼️ {self.images_dir.name} ({len(self.image_files)} imgs)"
        if self.labels_dir and self.labels_dir.exists():
            detect_icon = "🟢" if auto_detected else "🏷️"
            lbl_text += f"  |  {detect_icon} Labels: {self.labels_dir.name}"
        else:
            lbl_text += "  |  ⚠️ Labels directory not found"

        self.dir_status_lbl.config(text=lbl_text)

    # -------------------------------------------------------------------------
    # YAML & Class Definitions Parser
    # -------------------------------------------------------------------------
    def _parse_class_definitions(self, file_path: Path):
        """Parses classes.txt or YOLO dataset.yaml safely without external PyYAML dependency."""
        try:
            content = file_path.read_text(encoding="utf-8")
            new_classes: Dict[int, Dict[str, str]] = {}

            if file_path.suffix.lower() == ".txt":
                # Standard classes.txt: line 0 is class 0, line 1 is class 1, etc.
                for idx, line in enumerate(content.splitlines()):
                    name = line.strip()
                    if name and not name.startswith("#"):
                        color = self._get_color_for_id(idx)
                        new_classes[idx] = {"name": name, "color": color}
            else:
                # YAML format: Look for names: section
                # Patterns:
                # 1) names: {0: 'person', 1: 'face'}
                # 2) names: ['person', 'face']
                # 3) names:\n  0: person\n  1: face
                # 4) names:\n  - person\n  - face
                names_match = re.search(r"names\s*:\s*(.+)", content)
                if names_match:
                    inline_val = names_match.group(1).strip()
                    if inline_val.startswith("[") and inline_val.endswith("]"):
                        # Inline list: ['person', 'face']
                        items = re.findall(r"['\"]?([^'\",\[\]]+)['\"]?", inline_val)
                        for idx, item in enumerate(items):
                            name = item.strip()
                            if name:
                                new_classes[idx] = {"name": name, "color": self._get_color_for_id(idx)}
                    elif inline_val.startswith("{") and inline_val.endswith("}"):
                        # Inline dict: {0: 'person', 1: 'face'}
                        pairs = re.findall(r"(\d+)\s*:\s*['\"]?([^'\",{}]+)['\"]?", inline_val)
                        for cid_str, name in pairs:
                            cid = int(cid_str)
                            new_classes[cid] = {"name": name.strip(), "color": self._get_color_for_id(cid)}

                if not new_classes:
                    # Indented block under names:
                    in_names_block = False
                    list_idx = 0
                    for line in content.splitlines():
                        if re.match(r"^names\s*:", line):
                            in_names_block = True
                            continue
                        if in_names_block:
                            if line and not line.startswith(" ") and not line.startswith("\t"):
                                # End of indented block
                                break
                            # Key-value pattern: "  0: person"
                            kv_match = re.match(r"^\s*(\d+)\s*:\s*['\"]?([^'\"#\n]+)['\"]?", line)
                            if kv_match:
                                cid = int(kv_match.group(1))
                                name = kv_match.group(2).strip()
                                new_classes[cid] = {"name": name, "color": self._get_color_for_id(cid)}
                                continue
                            # Bullet pattern: "  - person"
                            bullet_match = re.match(r"^\s*-\s*['\"]?([^'\"#\n]+)['\"]?", line)
                            if bullet_match:
                                name = bullet_match.group(1).strip()
                                new_classes[list_idx] = {"name": name, "color": self._get_color_for_id(list_idx)}
                                list_idx += 1

            if new_classes:
                self.classes = new_classes
                # Ensure all classes have visibility variable
                for cid in self.classes.keys():
                    if cid not in self.class_visibility:
                        self.class_visibility[cid] = tk.BooleanVar(value=True)

        except Exception as exc:
            print(f"[WARNING] Error parsing class file {file_path}: {exc}")

    def _get_color_for_id(self, cid: int) -> str:
        """Returns pre-defined high-contrast color for default classes or picks from palette."""
        if cid in DEFAULT_CLASSES:
            return DEFAULT_CLASSES[cid]["color"]
        return EXTENDED_PALETTE[cid % len(EXTENDED_PALETTE)]

    # -------------------------------------------------------------------------
    # Image & Annotation Loading
    # -------------------------------------------------------------------------
    def _load_current_image(self):
        if not self.image_files or self.current_idx < 0 or self.current_idx >= len(self.image_files):
            self.current_pil_image = None
            self.current_bboxes = []
            self._render_empty_canvas("No images available in selected folder")
            self._update_status_bar()
            return

        img_path = self.image_files[self.current_idx]

        # 1. Load Image
        try:
            pil_img = Image.open(img_path)
            # Correct orientation from EXIF tags if present
            pil_img = ImageOps.exif_transpose(pil_img)
            self.current_pil_image = pil_img.convert("RGB")
        except Exception as exc:
            self.current_pil_image = None
            self.current_bboxes = []
            self._render_empty_canvas(f"Failed to load image:\n{img_path.name}\n({exc})")
            self._update_status_bar()
            return

        # 2. Load Corresponding Labels
        self.current_bboxes = self._load_yolo_labels_for_image(img_path)

        # 3. Reset zoom and pan for the new image
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # 4. Refresh Widgets & Render
        self._refresh_class_legend_widgets()
        self._update_status_bar()
        self._render_canvas()

    def _load_yolo_labels_for_image(self, img_path: Path) -> List[YoloBBox]:
        if not self.labels_dir or not self.labels_dir.exists():
            return []

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists() or not label_path.is_file():
            return []

        bboxes: List[YoloBBox] = []
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or len(parts) < 5:
                        continue
                    try:
                        cid = int(float(parts[0]))
                        xc = float(parts[1])
                        yc = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        conf = float(parts[5]) if len(parts) > 5 else None

                        # Validate that box dimensions are greater than zero
                        if w > 0.0 and h > 0.0:
                            bboxes.append(YoloBBox(cid, xc, yc, w, h, conf))

                            # Dynamically register any unknown class ID encountered
                            if cid not in self.classes:
                                self.classes[cid] = {
                                    "name": f"class_{cid}",
                                    "color": self._get_color_for_id(cid)
                                }
                                self.class_visibility[cid] = tk.BooleanVar(value=True)
                    except ValueError:
                        continue
        except Exception as exc:
            print(f"[WARNING] Error reading label file {label_path}: {exc}")

        return bboxes

    # -------------------------------------------------------------------------
    # Responsive Canvas Rendering & Zoom / Pan
    # -------------------------------------------------------------------------
    def _on_canvas_configure(self, event):
        """Debounced canvas configure to handle window resizing cleanly."""
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(35, self._render_canvas)

    def _render_empty_canvas(self, message: str):
        self.canvas.delete("all")
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        self.canvas.create_text(
            cw // 2, ch // 2,
            text=message,
            fill=THEME["text_secondary"],
            font=("Segoe UI", 12),
            justify=tk.CENTER
        )

    def _render_canvas(self):
        self._resize_job = None
        self.canvas.delete("all")

        if self.current_pil_image is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 10 or ch <= 10:
            return

        img_w, img_h = self.current_pil_image.size
        if img_w <= 0 or img_h <= 0:
            return

        # Base fit-to-window scale
        fit_scale = min(cw / img_w, ch / img_h)
        total_scale = fit_scale * self.zoom_level

        # Target rendered dimensions
        dw = max(1, int(img_w * total_scale))
        dh = max(1, int(img_h * total_scale))

        # Base origin to center the image
        base_ox = (cw - dw) / 2.0
        base_oy = (ch - dh) / 2.0

        # With user panning
        origin_x = base_ox + self.pan_x
        origin_y = base_oy + self.pan_y

        # Efficient visible crop for smooth rendering at any zoom factor
        vis_left = max(0.0, origin_x)
        vis_top = max(0.0, origin_y)
        vis_right = min(float(cw), origin_x + dw)
        vis_bottom = min(float(ch), origin_y + dh)

        if vis_right <= vis_left or vis_bottom <= vis_top:
            # Completely panned off screen
            return

        # Calculate crop coordinates in original image space
        crop_x1 = max(0.0, (vis_left - origin_x) / total_scale)
        crop_y1 = max(0.0, (vis_top - origin_y) / total_scale)
        crop_x2 = min(float(img_w), (vis_right - origin_x) / total_scale)
        crop_y2 = min(float(img_h), (vis_bottom - origin_y) / total_scale)

        crop_w = int(crop_x2 - crop_x1)
        crop_h = int(crop_y2 - crop_y1)

        dest_w = max(1, int((crop_x2 - crop_x1) * total_scale))
        dest_h = max(1, int((crop_y2 - crop_y1) * total_scale))

        try:
            cropped = self.current_pil_image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))
            # Use Bilinear for fast interactive panning, Lanczos when scale is < 1
            resample_filter = Image.Resampling.LANCZOS if total_scale < 1.0 else Image.Resampling.BILINEAR
            resized = cropped.resize((dest_w, dest_h), resample_filter)
            self.cached_tk_image = ImageTk.PhotoImage(resized)
            self.canvas.create_image(int(vis_left), int(vis_top), anchor="nw", image=self.cached_tk_image)
        except Exception as exc:
            print(f"[WARNING] Canvas rendering error: {exc}")
            return

        # ---------------------------------------------------------------------
        # Draw YOLO Bounding Boxes
        # ---------------------------------------------------------------------
        if self.show_bboxes:
            for bbox in self.current_bboxes:
                cid = bbox.class_id
                # Check class visibility toggle
                if cid in self.class_visibility and not self.class_visibility[cid].get():
                    continue

                class_info = self.classes.get(cid, {"name": f"class_{cid}", "color": "#00E5FF"})
                color = class_info["color"]
                name = class_info["name"]

                # Get pixel coordinates on original image
                orig_x1, orig_y1, orig_x2, orig_y2 = bbox.to_original_pixels(img_w, img_h)

                # Transform to canvas coordinates
                cx1 = origin_x + orig_x1 * total_scale
                cy1 = origin_y + orig_y1 * total_scale
                cx2 = origin_x + orig_x2 * total_scale
                cy2 = origin_y + orig_y2 * total_scale

                # Draw Bounding Box Rectangle
                box_width = 3 if total_scale > 1.5 else 2
                self.canvas.create_rectangle(
                    cx1, cy1, cx2, cy2,
                    outline=color,
                    width=box_width
                )

                # Draw Label Badge / Header
                if self.show_labels_text:
                    tag_text = f"{cid}: {name}"
                    if bbox.conf is not None:
                        tag_text += f" {bbox.conf:.2f}"

                    text_h = 16
                    text_w = len(tag_text) * 7 + 8

                    # Put tag above box if room, otherwise inside top of box
                    pill_y2 = cy1 if (cy1 - text_h) >= origin_y else cy1 + text_h
                    pill_y1 = pill_y2 - text_h if pill_y2 == cy1 else cy1
                    pill_x1 = cx1
                    pill_x2 = cx1 + text_w

                    # Tag background
                    self.canvas.create_rectangle(
                        pill_x1, pill_y1, pill_x2, pill_y2,
                        fill=color,
                        outline=color
                    )
                    # Text label (dark text for high contrast on bright vibrant tags)
                    self.canvas.create_text(
                        pill_x1 + 4, (pill_y1 + pill_y2) / 2.0,
                        anchor="w",
                        text=tag_text,
                        fill="#000000",
                        font=("Segoe UI", 9, "bold")
                    )

    # -------------------------------------------------------------------------
    # Mouse Zoom & Pan Handlers
    # -------------------------------------------------------------------------
    def _on_mouse_wheel(self, event):
        """Smooth zoom centered at current mouse pointer position."""
        if self.current_pil_image is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        img_w, img_h = self.current_pil_image.size
        fit_scale = min(cw / img_w, ch / img_h)

        old_zoom = self.zoom_level
        factor = 1.15 if event.delta > 0 else 0.85
        new_zoom = max(1.0, min(15.0, old_zoom * factor))

        if abs(new_zoom - old_zoom) < 0.001:
            return

        mouse_x = event.x
        mouse_y = event.y

        old_total_scale = fit_scale * old_zoom
        old_dw = img_w * old_total_scale
        old_dh = img_h * old_total_scale
        old_origin_x = (cw - old_dw) / 2.0 + self.pan_x
        old_origin_y = (ch - old_dh) / 2.0 + self.pan_y

        # Image point under cursor
        img_x = (mouse_x - old_origin_x) / old_total_scale
        img_y = (mouse_y - old_origin_y) / old_total_scale

        # Update zoom
        self.zoom_level = new_zoom
        new_total_scale = fit_scale * new_zoom
        new_dw = img_w * new_total_scale
        new_dh = img_h * new_total_scale
        new_base_ox = (cw - new_dw) / 2.0
        new_base_oy = (ch - new_dh) / 2.0

        if self.zoom_level <= 1.0:
            self.pan_x = 0.0
            self.pan_y = 0.0
        else:
            # Adjust pan so the same image point remains under mouse pointer
            new_origin_x = mouse_x - img_x * new_total_scale
            new_origin_y = mouse_y - img_y * new_total_scale
            self.pan_x = new_origin_x - new_base_ox
            self.pan_y = new_origin_y - new_base_oy

        self._render_canvas()

    def _zoom_by_factor(self, factor: float):
        if self.current_pil_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        # Fake event at center of canvas
        fake_event = type("Event", (), {"x": cw // 2, "y": ch // 2, "delta": 120 if factor > 1.0 else -120})()
        self._on_mouse_wheel(fake_event)

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._render_canvas()
        self.show_toast("Zoom reset (1.0x)")

    def _on_drag_start(self, event):
        self.is_dragging = True
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.canvas.config(cursor="fleur")

    def _on_drag_motion(self, event):
        if self.is_dragging and self.zoom_level > 1.0:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.pan_x += dx
            self.pan_y += dy
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self._render_canvas()

    def _on_drag_end(self, event):
        self.is_dragging = False
        self.canvas.config(cursor="crosshair")

    # -------------------------------------------------------------------------
    # Navigation & Jump Controls
    # -------------------------------------------------------------------------
    def next_image(self):
        if not self.image_files:
            return
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self._load_current_image()
        else:
            self.show_toast("Reached last image")

    def prev_image(self):
        if not self.image_files:
            return
        if self.current_idx > 0:
            self.current_idx -= 1
            self._load_current_image()
        else:
            self.show_toast("Reached first image")

    def go_to_first(self):
        if self.image_files and self.current_idx != 0:
            self.current_idx = 0
            self._load_current_image()

    def go_to_last(self):
        if self.image_files and self.current_idx != len(self.image_files) - 1:
            self.current_idx = len(self.image_files) - 1
            self._load_current_image()

    def _on_jump_return(self, event):
        val = self.jump_entry.get().strip()
        if not val.isdigit() or not self.image_files:
            return
        target = int(val) - 1  # 1-indexed to 0-indexed
        if 0 <= target < len(self.image_files):
            self.current_idx = target
            self._load_current_image()
        else:
            self.show_toast(f"Invalid index: 1 to {len(self.image_files)}")

    # -------------------------------------------------------------------------
    # Audit Actions (Flag & Trash)
    # -------------------------------------------------------------------------
    def _load_flagged_list(self):
        self.flagged_files.clear()
        if self.flagged_log_path and self.flagged_log_path.exists():
            try:
                with open(self.flagged_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        name = line.strip()
                        if name:
                            self.flagged_files.add(name)
            except Exception as exc:
                print(f"[WARNING] Error reading flagged file: {exc}")

    def toggle_flag_current_image(self):
        if not self.image_files or self.current_idx < 0:
            return

        current_file = self.image_files[self.current_idx].name
        if not self.flagged_log_path:
            self.flagged_log_path = self.images_dir.parent / "flagged_for_review.txt"

        if current_file in self.flagged_files:
            # Unflag
            self.flagged_files.remove(current_file)
            self._save_flagged_list()
            self.show_toast(f"Unflagged: {current_file}")
        else:
            # Flag
            self.flagged_files.add(current_file)
            self._save_flagged_list()
            self.show_toast(f"🚩 Flagged: {current_file}")

        self._update_status_bar()

    def _save_flagged_list(self):
        if not self.flagged_log_path:
            return
        try:
            with open(self.flagged_log_path, "w", encoding="utf-8") as f:
                for name in sorted(self.flagged_files):
                    f.write(f"{name}\n")
        except Exception as exc:
            messagebox.showerror("Error Saving Flag Log", str(exc))

    def trash_current_label(self):
        """Moves the corresponding .txt label file to a trash/ subdirectory."""
        if not self.image_files or self.current_idx < 0 or not self.labels_dir:
            return

        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            self.show_toast("No label file exists for this image")
            return

        trash_dir = self.labels_dir / "trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        dest_trash_path = trash_dir / label_path.name

        try:
            # Move file
            shutil.move(str(label_path), str(dest_trash_path))
            self.last_trashed_info = (dest_trash_path, label_path)
            self.current_bboxes = []
            self._refresh_class_legend_widgets()
            self._update_status_bar()
            self._render_canvas()
            self.show_toast(f"🗑️ Moved {label_path.name} to labels/trash/")
        except Exception as exc:
            messagebox.showerror("Error Moving Label to Trash", str(exc))

    def undo_trash_label(self):
        """Restores the last trashed label file."""
        if not self.last_trashed_info:
            self.show_toast("No trash action to undo")
            return

        trash_path, original_path = self.last_trashed_info
        if not trash_path.exists():
            self.show_toast("Trash file no longer exists")
            return

        try:
            shutil.move(str(trash_path), str(original_path))
            self.last_trashed_info = None
            self._load_current_image()
            self.show_toast(f"↩️ Restored {original_path.name}")
        except Exception as exc:
            messagebox.showerror("Error Undoing Trash", str(exc))

    def open_in_file_explorer(self):
        if not self.images_dir or not self.images_dir.exists():
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(self.images_dir))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(self.images_dir)])
        except Exception as exc:
            messagebox.showwarning("Cannot Open Folder", str(exc))

    # -------------------------------------------------------------------------
    # UI Visibility Toggles
    # -------------------------------------------------------------------------
    def _on_class_visibility_toggled(self):
        self._render_canvas()

    def show_all_classes(self):
        for var in self.class_visibility.values():
            var.set(True)
        self._render_canvas()

    def hide_all_classes(self):
        for var in self.class_visibility.values():
            var.set(False)
        self._render_canvas()

    def toggle_label_text(self):
        self.show_labels_text = not self.show_labels_text
        self._render_canvas()
        self.show_toast(f"Label text: {'Shown' if self.show_labels_text else 'Hidden'}")

    def toggle_bboxes(self):
        self.show_bboxes = not self.show_bboxes
        self._render_canvas()
        self.show_toast(f"Bounding boxes: {'Shown' if self.show_bboxes else 'Hidden'}")

    # -------------------------------------------------------------------------
    # Status Updates & Feedback Toasts
    # -------------------------------------------------------------------------
    def _update_status_bar(self):
        total = len(self.image_files)
        if total == 0 or self.current_idx < 0:
            self.lbl_counter.config(text="Image: 0 / 0")
            self.lbl_filename.config(text="No file selected")
            self.lbl_dimensions.config(text="-- × --")
            self.lbl_objects.config(text="Objects: 0")
            self.flag_badge.pack_forget()
            return

        current_file = self.image_files[self.current_idx]
        self.lbl_counter.config(text=f"Image: {self.current_idx + 1} / {total}")
        self.lbl_filename.config(text=current_file.name)

        # Flag indicator
        if current_file.name in self.flagged_files:
            self.flag_badge.pack(side=tk.LEFT, padx=6)
            self.btn_flag.config(text="🚩 Unflag Image (F)")
        else:
            self.flag_badge.pack_forget()
            self.btn_flag.config(text="🚩 Flag Current Image (F)")

        if self.current_pil_image:
            w, h = self.current_pil_image.size
            self.lbl_dimensions.config(text=f"{w} × {h} px")
        else:
            self.lbl_dimensions.config(text="-- × --")

        # Object count summary
        num_boxes = len(self.current_bboxes)
        self.lbl_objects.config(text=f"Total Objects: {num_boxes}")

    def show_toast(self, message: str, duration_ms: int = 2500):
        """Displays a clean floating toast notification over the canvas."""
        if self.toast_job is not None:
            self.root.after_cancel(self.toast_job)

        self.toast_label.config(text=message)
        self.toast_label.place(relx=0.5, rely=0.06, anchor="center")
        self.toast_job = self.root.after(duration_ms, self._hide_toast)

    def _hide_toast(self):
        self.toast_label.place_forget()
        self.toast_job = None

    def show_shortcuts_dialog(self):
        msg = """⌨️ Keyboard & Mouse Shortcuts:

[Navigation]
  • D  / Right Arrow : Next Image
  • A  / Left Arrow  : Previous Image
  • Home             : First Image
  • End              : Last Image

[Auditing Actions]
  • F                : Flag / Unflag Image (saves to flagged_for_review.txt)
  • Delete           : Move current label file to labels/trash/
  • Ctrl + Z         : Undo last trashed label file

[Zoom & View]
  • Mouse Wheel      : Smooth Zoom In / Out at Cursor
  • Left-Click Drag  : Pan Canvas when Zoomed In
  • Double-Click / R : Reset Zoom to Fit
  • +  /  -          : Zoom In / Out by Step
  • T                : Toggle Class Label Text & Pills
  • B                : Toggle All Bounding Boxes
  • F1  /  ?         : Show this Shortcuts Reference
"""
        messagebox.showinfo("YOLO Auditor Shortcuts", msg)


# =============================================================================
# Application Entry Point
# =============================================================================
def main():
    root = tk.Tk()
    app = YoloDatasetAuditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
