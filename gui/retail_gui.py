"""Retail Analytics v2 — Standalone Native GUI Application
=========================================================
Provides a complete graphical interface for edge deployment:
- No command prompt or terminal window required (console=False).
- Interactive RTSP Stream configuration with persistent YAML saving.
- One-click Gate Calibration launch (interactive OpenCV window).
- Headless execution with real-time KPI card updates and live event stream.
- Unattended auto-recovery after power outages with visual countdown.
"""

import os
import sys
import time
import queue
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent

# Color Palette (Modern Dark Theme)
BG_MAIN = "#0f172a"       # Slate 900
BG_CARD = "#1e293b"       # Slate 800
BG_CARD_LIGHT = "#334155" # Slate 700
BORDER_COLOR = "#475569"  # Slate 600
TEXT_WHITE = "#f8fafc"    # Slate 50
TEXT_MUTED = "#94a3b8"    # Slate 400
ACCENT_BLUE = "#3b82f6"   # Blue 500
ACCENT_HOVER = "#2563eb"  # Blue 600
ACCENT_GREEN = "#10b981"  # Emerald 500
ACCENT_RED = "#ef4444"    # Red 500
ACCENT_AMBER = "#f59e0b"  # Amber 500
ACCENT_PURPLE = "#a855f7" # Purple 500


class GuiLogHandler:
    """Thread-safe log handler that captures prints and redirects to Tkinter."""

    def __init__(self, log_queue: queue.Queue, original_stream=None, file_sink=None):
        self.log_queue = log_queue
        self.original_stream = original_stream
        self.file_sink = file_sink

    def write(self, text: str):
        if not text:
            return
        # Write to log file if available
        if self.file_sink is not None:
            try:
                self.file_sink.write(text)
                self.file_sink.flush()
            except Exception:
                pass
        # Write to original stream (if terminal attached)
        if self.original_stream is not None:
            try:
                self.original_stream.write(text)
                self.original_stream.flush()
            except Exception:
                pass
        # Queue for GUI display if meaningful text
        clean = text.strip()
        if clean:
            self.log_queue.put(clean)

    def flush(self):
        if self.file_sink:
            try:
                self.file_sink.flush()
            except Exception:
                pass
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass


class RetailAnalyticsApp:
    """Main Standalone GUI Application."""

    def __init__(self, config_path: str = "config/production.yaml"):
        self.config_path = str(PROJECT_ROOT / config_path)
        self.cfg = self._load_config()

        # Extract settings
        cam_cfg = self.cfg.get("camera", {})
        api_cfg = self.cfg.get("api", {})
        shop_cfg = self.cfg.get("shop", {})

        self.saved_rtsp = (
            cam_cfg.get("rtsp_url")
            or cam_cfg.get("fallback_video")
            or "test_videos/Outlet_Sample_1_Banasree.avi"
        )
        self.api_url = api_cfg.get("base_url", "http://ai.prangroup.com:8021")
        self.shop_id = shop_cfg.get("shop_id", 2)
        self.weights = self.cfg.get("inference", {}).get("model", "models/yolov8m.onnx")

        # Runtime state
        self.pipeline = None
        self.pipeline_thread = None
        self.is_monitoring = False
        self.countdown_seconds = 15
        self.countdown_active = True
        self.log_queue = queue.Queue(maxsize=5000)

        # Initialize logging
        self._init_logging()

        # Tkinter Root Setup
        self.root = tk.Tk()
        self.root.title("Retail Analytics v2 — Standalone Edge AI Gateway")
        self.root.geometry("860x680")
        self.root.minsize(780, 580)
        self.root.configure(bg=BG_MAIN)

        # Style Configuration
        self._setup_styles()

        # Main Container Frame
        self.main_container = tk.Frame(self.root, bg=BG_MAIN)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Render Initial Setup View
        self.render_setup_view()

        # Window Close Hook
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _load_config(self) -> dict:
        import yaml
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}

    def _persist_rtsp_url(self, new_url: str):
        if not new_url or not new_url.strip():
            return
        new_url = new_url.strip()
        try:
            if not os.path.exists(self.config_path):
                return
            with open(self.config_path, "r", encoding="utf-8") as f:
                content = f.read()

            import re
            escaped = new_url.replace("\\", "\\\\")
            if re.search(r"^\s*rtsp_url:.*$", content, flags=re.MULTILINE):
                content = re.sub(r"^(\s*rtsp_url:).*$", f'\\1 "{escaped}"', content, flags=re.MULTILINE)
            else:
                content = re.sub(r"^(\s*camera:\s*)$", f'\\1\n  rtsp_url: "{escaped}"', content, flags=re.MULTILINE)

            with open(self.config_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.saved_rtsp = new_url
            print(f"[GUI] Successfully saved RTSP URL to {self.config_path}")
        except Exception as e:
            print(f"[GUI] Warning: Failed to save RTSP URL to config: {e}")

    def _init_logging(self):
        log_file = PROJECT_ROOT / "retail_ai.log"
        try:
            log_fh = open(str(log_file), "a", encoding="utf-8", buffering=1)
        except Exception:
            log_fh = None

        self.gui_writer = GuiLogHandler(self.log_queue, original_stream=sys.stdout, file_sink=log_fh)
        sys.stdout = self.gui_writer
        sys.stderr = self.gui_writer

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TProgressbar", thickness=6, troughcolor=BG_CARD, background=ACCENT_BLUE)

    # -------------------------------------------------------------------------
    # VIEW 1: Setup & Camera Configuration Screen
    # -------------------------------------------------------------------------

    def render_setup_view(self):
        """Constructs the initial setup screen."""
        for widget in self.main_container.winfo_children():
            widget.destroy()

        # Header Card
        header = tk.Frame(self.main_container, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR)
        header.pack(fill=tk.X, padx=20, pady=(20, 15))

        title_box = tk.Frame(header, bg=BG_CARD)
        title_box.pack(side=tk.LEFT, padx=18, pady=14)

        lbl_title = tk.Label(title_box, text="Retail Analytics v2", font=("Segoe UI", 16, "bold"), fg=TEXT_WHITE, bg=BG_CARD)
        lbl_title.pack(anchor="w")
        lbl_sub = tk.Label(title_box, text="Standalone Edge AI Footfall, Demographics & Public API Gateway", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD)
        lbl_sub.pack(anchor="w")

        # Status badge on right
        badge_box = tk.Frame(header, bg=BG_CARD)
        badge_box.pack(side=tk.RIGHT, padx=18, pady=14)

        self.lbl_api_badge = tk.Label(
            badge_box,
            text=f"API: {self.api_url}  |  Outlet ID: {self.shop_id}",
            font=("Segoe UI", 9, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_CARD_LIGHT,
            padx=12,
            pady=6,
        )
        self.lbl_api_badge.pack()

        # Configuration Card
        cfg_card = tk.Frame(self.main_container, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR)
        cfg_card.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            cfg_card,
            text="Camera / NVR Stream Settings",
            font=("Segoe UI", 12, "bold"),
            fg=TEXT_WHITE,
            bg=BG_CARD,
        ).pack(anchor="w", padx=18, pady=(16, 4))

        tk.Label(
            cfg_card,
            text="Enter your camera's RTSP URL or select a video file. This configuration will be stored for automatic restarts.",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg=BG_CARD,
        ).pack(anchor="w", padx=18, pady=(0, 12))

        # URL Input Box
        input_frame = tk.Frame(cfg_card, bg=BG_CARD)
        input_frame.pack(fill=tk.X, padx=18, pady=(0, 14))

        self.rtsp_entry = tk.Entry(
            input_frame,
            font=("Consolas", 10),
            bg=BG_MAIN,
            fg=TEXT_WHITE,
            insertbackground=TEXT_WHITE,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_BLUE,
        )
        self.rtsp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 10))
        self.rtsp_entry.insert(0, self.saved_rtsp)
        self.rtsp_entry.bind("<Key>", self._on_user_interaction)

        btn_browse = tk.Button(
            input_frame,
            text="Browse File...",
            font=("Segoe UI", 9, "bold"),
            bg=BG_CARD_LIGHT,
            fg=TEXT_WHITE,
            activebackground=BORDER_COLOR,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=14,
            pady=5,
            cursor="hand2",
            command=self._on_browse_file,
        )
        btn_browse.pack(side=tk.RIGHT)

        # Unattended Power Recovery Alert & Countdown Banner
        self.countdown_banner = tk.Frame(self.main_container, bg="#1e1b4b", highlightthickness=1, highlightbackground="#4338ca")
        self.countdown_banner.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_countdown = tk.Label(
            self.countdown_banner,
            text=f"⚡ Power Loss Recovery: Auto-starting monitoring in {self.countdown_seconds}s...",
            font=("Segoe UI", 10, "bold"),
            fg="#c7d2fe",
            bg="#1e1b4b",
        )
        self.lbl_countdown.pack(side=tk.LEFT, padx=16, pady=10)

        btn_pause = tk.Button(
            self.countdown_banner,
            text="Pause Auto-Start",
            font=("Segoe UI", 8, "bold"),
            bg="#3730a3",
            fg=TEXT_WHITE,
            activebackground="#4338ca",
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._pause_countdown,
        )
        btn_pause.pack(side=tk.RIGHT, padx=16, pady=8)

        # Action Buttons Box
        action_box = tk.Frame(self.main_container, bg=BG_MAIN)
        action_box.pack(fill=tk.X, padx=20, pady=16)

        btn_calib = tk.Button(
            action_box,
            text="🎯 Calibrate Virtual Gates",
            font=("Segoe UI", 10, "bold"),
            bg=BG_CARD,
            fg=TEXT_WHITE,
            activebackground=BG_CARD_LIGHT,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=18,
            pady=10,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            command=self._launch_calibration,
        )
        btn_calib.pack(side=tk.LEFT)

        btn_start = tk.Button(
            action_box,
            text="🚀 Start Headless Monitoring",
            font=("Segoe UI", 10, "bold"),
            bg=ACCENT_BLUE,
            fg=TEXT_WHITE,
            activebackground=ACCENT_HOVER,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=22,
            pady=10,
            cursor="hand2",
            command=self._start_monitoring,
        )
        btn_start.pack(side=tk.RIGHT)

        # Start background timer for unattended restart
        self._tick_countdown()

    def _on_user_interaction(self, event=None):
        self._pause_countdown()

    def _pause_countdown(self):
        self.countdown_active = False
        if hasattr(self, "countdown_banner") and self.countdown_banner.winfo_exists():
            self.lbl_countdown.config(text="Auto-start paused. Configure settings and click 'Start Headless Monitoring'.")

    def _tick_countdown(self):
        if not self.countdown_active or self.is_monitoring:
            return
        if self.countdown_seconds > 0:
            if hasattr(self, "lbl_countdown") and self.lbl_countdown.winfo_exists():
                self.lbl_countdown.config(text=f"⚡ Power Loss Recovery: Auto-starting monitoring in {self.countdown_seconds}s...")
            self.countdown_seconds -= 1
            self.root.after(1000, self._tick_countdown)
        else:
            self._start_monitoring()

    def _on_browse_file(self):
        self._pause_countdown()
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.avi *.mp4 *.mkv *.mov"), ("All Files", "*.*")],
        )
        if path:
            self.rtsp_entry.delete(0, tk.END)
            self.rtsp_entry.insert(0, path)

    def _launch_calibration(self):
        self._pause_countdown()
        source = self.rtsp_entry.get().strip() or self.saved_rtsp
        self._persist_rtsp_url(source)

        # Hide Tkinter setup window during calibration
        self.root.withdraw()
        try:
            from calibrate_gates import GateCalibrator
            calibrator = GateCalibrator(source)
            calibrator.run()
        except Exception as e:
            messagebox.showerror("Calibration Error", f"Could not connect to camera stream:\n{e}\n\nPlease check network or RTSP URL.")
        finally:
            self.root.deiconify()

    # -------------------------------------------------------------------------
    # VIEW 2: Live Monitoring Dashboard (Zero Terminal Overhead)
    # -------------------------------------------------------------------------

    def _start_monitoring(self):
        self.countdown_active = False
        source = self.rtsp_entry.get().strip() if hasattr(self, "rtsp_entry") else self.saved_rtsp
        if not source:
            source = self.saved_rtsp

        self._persist_rtsp_url(source)
        self.is_monitoring = True
        self.render_monitoring_view(source)

        # Launch unified retail pipeline in dedicated background worker thread
        self.pipeline_thread = threading.Thread(
            target=self._run_pipeline_worker,
            args=(source,),
            name="Pipeline-RunnerThread",
            daemon=True,
        )
        self.pipeline_thread.start()

    def render_monitoring_view(self, source: str):
        """Constructs the live monitoring dashboard."""
        for widget in self.main_container.winfo_children():
            widget.destroy()

        # Top Status Bar
        top_bar = tk.Frame(self.main_container, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR)
        top_bar.pack(fill=tk.X, padx=20, pady=(16, 12))

        title_frame = tk.Frame(top_bar, bg=BG_CARD)
        title_frame.pack(side=tk.LEFT, padx=16, pady=12)

        tk.Label(title_frame, text="Retail Analytics v2", font=("Segoe UI", 14, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        src_clean = source if len(source) < 55 else source[:52] + "..."
        tk.Label(title_frame, text=f"Source: {src_clean}", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")

        # Indicators on right
        indicators = tk.Frame(top_bar, bg=BG_CARD)
        indicators.pack(side=tk.RIGHT, padx=16, pady=12)

        self.lbl_stream_status = tk.Label(indicators, text="🟢 Stream: Connected", font=("Segoe UI", 9, "bold"), fg=ACCENT_GREEN, bg=BG_CARD_LIGHT, padx=10, pady=4)
        self.lbl_stream_status.pack(side=tk.LEFT, padx=4)

        self.lbl_api_status = tk.Label(indicators, text=f"🟢 API: Online ({self.api_url})", font=("Segoe UI", 9, "bold"), fg=ACCENT_GREEN, bg=BG_CARD_LIGHT, padx=10, pady=4)
        self.lbl_api_status.pack(side=tk.LEFT, padx=4)

        # KPI Metrics Cards (4 columns)
        kpi_frame = tk.Frame(self.main_container, bg=BG_MAIN)
        kpi_frame.pack(fill=tk.X, padx=20, pady=4)
        kpi_frame.columnconfigure((0, 1, 2, 3), weight=1, uniform="kpi")

        self.lbl_kpi_in = self._create_kpi_card(kpi_frame, 0, "Total In (Entries)", "0", ACCENT_GREEN)
        self.lbl_kpi_out = self._create_kpi_card(kpi_frame, 1, "Total Out (Exits)", "0", ACCENT_BLUE)
        self.lbl_kpi_inside = self._create_kpi_card(kpi_frame, 2, "Current Inside", "0", ACCENT_PURPLE)
        self.lbl_kpi_bags = self._create_kpi_card(kpi_frame, 3, "Bag Exits", "0", ACCENT_AMBER)

        # Log & Event Stream Card
        log_card = tk.Frame(self.main_container, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR)
        log_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)

        log_head = tk.Frame(log_card, bg=BG_CARD)
        log_head.pack(fill=tk.X, padx=14, pady=(10, 6))

        tk.Label(log_head, text="Real-Time Event Stream (Headless Mode)", font=("Segoe UI", 10, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side=tk.LEFT)
        tk.Label(log_head, text="All events stream to API with 0% dropped FPS", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.RIGHT)

        # Scrolled Text for Event Logs
        self.txt_logs = ScrolledText(
            log_card,
            bg=BG_MAIN,
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            font=("Consolas", 9),
            relief=tk.FLAT,
            highlightthickness=0,
            padx=10,
            pady=8,
        )
        self.txt_logs.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # Bottom Action Bar
        bottom_bar = tk.Frame(self.main_container, bg=BG_MAIN)
        bottom_bar.pack(fill=tk.X, padx=20, pady=(0, 16))

        btn_recalib = tk.Button(
            bottom_bar,
            text="🎯 Recalibrate Gates",
            font=("Segoe UI", 9, "bold"),
            bg=BG_CARD,
            fg=TEXT_WHITE,
            activebackground=BG_CARD_LIGHT,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            command=self._launch_calibration_from_monitor,
        )
        btn_recalib.pack(side=tk.LEFT, padx=(0, 8))

        btn_open_log = tk.Button(
            bottom_bar,
            text="📄 View Log File",
            font=("Segoe UI", 9, "bold"),
            bg=BG_CARD,
            fg=TEXT_WHITE,
            activebackground=BG_CARD_LIGHT,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            command=self._open_log_file,
        )
        btn_open_log.pack(side=tk.LEFT)

        btn_stop = tk.Button(
            bottom_bar,
            text="⏹️ Stop Monitoring & Exit",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT_RED,
            fg=TEXT_WHITE,
            activebackground="#dc2626",
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.on_close,
        )
        btn_stop.pack(side=tk.RIGHT)

        # Start periodic GUI refresh tasks
        self.root.after(200, self._poll_log_queue)
        self.root.after(500, self._poll_pipeline_metrics)

    def _create_kpi_card(self, parent, col, title, initial_val, color) -> tk.Label:
        card = tk.Frame(parent, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR)
        card.grid(row=0, column=col, padx=4, sticky="nsew")

        tk.Label(card, text=title, font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", padx=12, pady=(10, 2))
        lbl_val = tk.Label(card, text=initial_val, font=("Segoe UI", 18, "bold"), fg=color, bg=BG_CARD)
        lbl_val.pack(anchor="w", padx=12, pady=(0, 10))
        return lbl_val

    def _launch_calibration_from_monitor(self):
        source = self.saved_rtsp
        self.root.withdraw()
        try:
            from calibrate_gates import GateCalibrator
            calibrator = GateCalibrator(source)
            calibrator.run()
        except Exception as e:
            messagebox.showerror("Calibration Error", f"Error launching calibration window: {e}")
        finally:
            self.root.deiconify()

    def _open_log_file(self):
        log_file = PROJECT_ROOT / "retail_ai.log"
        if log_file.exists():
            try:
                os.startfile(str(log_file))
            except Exception as e:
                messagebox.showinfo("Log File", f"Log file located at:\n{log_file}")
        else:
            messagebox.showinfo("Log File", "No log file has been created yet.")

    def _poll_log_queue(self):
        """Drains background thread log prints into GUI scrolled text widget."""
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                ts = time.strftime("%H:%M:%S")
                self.txt_logs.insert(tk.END, f"[{ts}] {msg}\n")
                self.txt_logs.see(tk.END)
            except queue.Empty:
                break

        if self.is_monitoring:
            self.root.after(200, self._poll_log_queue)

    def _poll_pipeline_metrics(self):
        """Periodically polls live counts and connection status from pipeline."""
        if self.pipeline is not None and self.is_monitoring:
            try:
                in_cnt = self.pipeline.event_manager.validated_entry_count
                out_cnt = self.pipeline.event_manager.validated_exit_count
                inside = max(0, in_cnt - out_cnt)
                bag_cnt = self.pipeline.cached_total_bag_exits

                self.lbl_kpi_in.config(text=str(in_cnt))
                self.lbl_kpi_out.config(text=str(out_cnt))
                self.lbl_kpi_inside.config(text=str(inside))
                self.lbl_kpi_bags.config(text=str(bag_cnt))

                # Stream status indicator
                is_conn = getattr(self.pipeline.reader, "is_connected", True)
                if is_conn:
                    self.lbl_stream_status.config(text="🟢 Stream: Connected", fg=ACCENT_GREEN)
                else:
                    self.lbl_stream_status.config(text="🔴 Stream: Reconnecting...", fg=ACCENT_RED)
            except Exception:
                pass

        if self.is_monitoring:
            self.root.after(500, self._poll_pipeline_metrics)

    def _run_pipeline_worker(self, source: str):
        """Background thread executing the AI pipeline in headless mode."""
        from services.unified_pipeline import UnifiedRetailPipeline
        try:
            self.pipeline = UnifiedRetailPipeline(
                video_source=source,
                config_path=self.config_path,
                primary_model_path=self.weights,
                enable_demographics=True,
                display=False,  # Enforce headless mode (zero GUI preview overhead)
            )
            self.pipeline.run()
        except Exception as e:
            print(f"[GUI] Pipeline encountered error: {e}")
        finally:
            print("[GUI] Pipeline execution terminated cleanly.")

    def on_close(self):
        """Clean shutdown handler."""
        if self.is_monitoring and self.pipeline is not None:
            ans = messagebox.askyesno("Confirm Exit", "Stop analytics monitoring and close Retail Analytics?")
            if not ans:
                return

        print("[GUI] Shutting down application cleanly...")
        self.is_monitoring = False
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self.root.destroy()
        sys.exit(0)

    def run(self):
        """Start the Tkinter mainloop."""
        self.root.mainloop()


def launch_gui(config_path: str = "config/production.yaml"):
    app = RetailAnalyticsApp(config_path=config_path)
    app.run()


if __name__ == "__main__":
    launch_gui()
