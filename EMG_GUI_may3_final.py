# -*- coding: utf-8 -*-
"""
ReSync — EMG Activation Monitor
=================================
Real hardware: Arduino Uno + MyoWare 2.0 + HC-05 Bluetooth.

Install dependencies:
    pip install matplotlib numpy pyserial

Run:
    python EMG_GUI.py
"""

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ─────────────────────────────────────────────
# OPENAI API KEY  — paste your key here
# ─────────────────────────────────────────────

OPENAI_API_KEY = ""

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

COLORS = {
    "bg":       "#0d0d14",
    "surface":  "#13131f",
    "surface2": "#1a1a2e",
    "border":   "#2a2a45",
    "accent":   "#7c6af7",
    "accent2":  "#4fc3f7",
    "green":    "#4ade80",
    "yellow":   "#fbbf24",
    "red":      "#f87171",
    "text":     "#e8e8f0",
    "text2":    "#8888aa",
    "text3":    "#4a4a6a",
    "plot_bg":  "#0a0a12",
    "dot_empty":"#2a2a45",
    "dot_pulse":"#ffffff",
}

# Rep dot colors by progress zone
DOT_COLORS = [
    "#4fc3f7",  # 1-7:  cool blue
    "#4fc3f7",
    "#4fc3f7",
    "#4fc3f7",
    "#4fc3f7",
    "#4fc3f7",
    "#4fc3f7",
    "#4ade80",  # 8-14: green
    "#4ade80",
    "#4ade80",
    "#4ade80",
    "#4ade80",
    "#4ade80",
    "#4ade80",
    "#fbbf24",  # 15-20: yellow/orange getting intense
    "#f59e0b",
    "#f97316",
    "#fb923c",
    "#ef4444",
    "#dc2626",
]

FONT_TITLE = ("Courier New", 28, "bold")
FONT_LABEL = ("Courier New", 9)
FONT_BIG   = ("Courier New", 48, "bold")
FONT_MED   = ("Courier New", 20, "bold")
FONT_SMALL = ("Courier New", 10)
FONT_MONO  = ("Courier New", 11)
FONT_BTN   = ("Courier New", 11, "bold")

HISTORY_LEN    = 300
WARMUP_SAMPLES = 200
FATIGUE_RATIO  = 1.4
Z_THRESH       = 6.0    # raised to suit low-baseline signal
Z_DEACT_THRESH = 3.5    # deactivation threshold
Z_SCORE_SCALE  = 200.0
MAX_REPS       = 20
CONSEC_FATIGUE = 3   # consecutive reps above threshold to flag fatigue

# ─────────────────────────────────────────────
# FABRICATED HISTORICAL DATA
# ─────────────────────────────────────────────

np.random.seed(42)

_base_fatigue = np.linspace(4.5, 11.0, 10)
HISTORICAL_FATIGUE_ONSET = np.clip(
    _base_fatigue + np.random.normal(0, 0.9, 10), 2, 15
).astype(int).tolist()

_base_hold = np.linspace(2.0, 3.5, 10)
HISTORICAL_HOLD_TIME = np.clip(
    _base_hold + np.random.normal(0, 0.25, 10), 1.2, 4.5
).round(2).tolist()

SESSION_LABELS = [f"S{i+1}" for i in range(10)]

# ─────────────────────────────────────────────
# BLUETOOTH READER
# ─────────────────────────────────────────────

class BluetoothReader:
    def __init__(self, store, port, baud=9600):
        self.store   = store
        self.port    = port
        self.baud    = baud
        self.ser     = None
        self.running = False

    def start(self):
        self.ser     = serial.Serial(self.port, self.baud, timeout=1, write_timeout=2)
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, text):
        threading.Thread(target=self._send, args=(text,), daemon=True).start()

    def _send(self, text):
        try:
            if self.ser and self.ser.is_open:
                self.ser.write((str(text) + '\n').encode('utf-8'))
        except Exception:
            pass

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _run(self):
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8', errors='replace').strip()
                if line:
                    print(f"[RX] {line}")
                    self._parse(line)
            except Exception:
                pass

    def _parse(self, line):
        s = self.store

        # ── Main telemetry ──
        if line.startswith('raw='):
            data = {}
            for token in line.split():
                if '=' in token:
                    k, _, v = token.partition('=')
                    data[k] = v
            try:
                s.emg_raw = int(data['raw'])
                sm = int(data['sm'])
                s.emg_history.append(sm)
                if len(s.emg_history) > HISTORY_LEN:
                    s.emg_history.pop(0)
                s.z_score = float(data['z'])

                for k in data:
                    if k in ('μ', 'u', 'mu'):
                        s.mu = float(data[k])
                    if k in ('σ', 's', 'sigma'):
                        s.sigma = float(data[k])

                new_active = (data.get('act') == 'YES')

                # Rep hold time tracking
                if new_active and not s.active:
                    s._rep_start_time = time.time()
                elif not new_active and s.active and s._rep_start_time is not None:
                    duration = time.time() - s._rep_start_time
                    if 0.1 < duration < 10.0:
                        s.rep_durations.append(round(duration, 2))
                    s._rep_start_time = None

                s.active = new_active
            except (KeyError, ValueError):
                pass

        # ── Warmup progress ──
        elif line.startswith('Warmup'):
            try:
                nums = line.split()[1].split('/')
                s.cal_progress = int(nums[0]) / int(nums[1])
            except (IndexError, ValueError, ZeroDivisionError):
                pass

        # ── Baseline ready ──
        elif 'Baseline ready' in line or 'run until fatigue' in line:
            s.cal_progress  = 1.0
            s.arduino_ready = True

        # ── Rep count ──
        elif line.startswith('Rep:'):
            try:
                new_count = int(line.split(':')[1].strip())
                if new_count > s.rep_count:
                    s.rep_count      = new_count
                    s.new_rep_pulse  = new_count   # signal pulse animation
            except (ValueError, IndexError):
                pass

        # ── Peak z per rep — 3-consecutive fatigue detection ──
        elif line.startswith('peakZ='):
            data = {}
            for token in line.split():
                if '=' in token:
                    k, _, v = token.partition('=')
                    data[k] = v
            try:
                peak = float(data['peakZ'])
                base = float(data['baseline'])
                s.rep_peaks.append(peak)

                # Rolling window for consecutive fatigue detection
                s.fatigue_window.append(peak)
                if len(s.fatigue_window) > CONSEC_FATIGUE:
                    s.fatigue_window.pop(0)

                if (len(s.fatigue_window) == CONSEC_FATIGUE
                        and base > 0
                        and all(p > base * FATIGUE_RATIO
                                for p in s.fatigue_window)):
                    if s.fatigue_onset_rep is None:
                        s.fatigue_onset_rep   = s.rep_count
                        s.session_complete    = True
                        s.trigger_fatigue_led = True  # tell Arduino to light LED

            except (KeyError, ValueError):
                pass

        # ── Session stop conditions ──
        elif 'FATIGUE DETECTED' in line:
            if s.fatigue_onset_rep is None:
                s.fatigue_onset_rep = s.rep_count
            s.session_complete = True

        elif 'TARGET REPS REACHED' in line:
            s.session_complete = True

        elif 'Session ended early' in line:
            s.session_complete = True

        s.last_rx       = time.time()
        s.ever_received = True


# ─────────────────────────────────────────────
# APP STATE
# ─────────────────────────────────────────────

class Store:
    def __init__(self):
        self.screen            = "CONNECT"
        self.connected         = False
        self.emg_raw           = 0
        self.emg_history       = []
        self.z_score           = 0.0
        self.mu                = 22.0
        self.sigma             = 3.4
        self.active            = False
        self.rep_count         = 0
        self.rep_peaks         = []
        self.rep_durations     = []
        self.fatigue_window      = []   # rolling window for consecutive detection
        self.fatigue_onset_rep   = None
        self.trigger_fatigue_led = False
        self.new_rep_pulse     = 0    # signals GUI to animate dot N
        self.cal_progress      = 0.0
        self.session_start     = None
        self.metrics           = {}
        self.arduino_ready     = False
        self.session_complete  = False
        self.target_reps       = 0
        self.port_open         = False
        self.last_rx           = 0.0
        self.ever_received     = False
        self._rep_start_time   = None


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

class ReSync(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ReSync")
        self.geometry("1000x700")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        self.store  = Store()
        self.reader = None

        self._cal_ready_shown = False
        self._session_ended   = False
        self._after_id        = None
        self._last_pulsed_rep = 0     # track which rep we last animated
        self._dot_ids         = []    # canvas oval IDs
        self._dot_pulsing     = False

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._update_loop()

    # ─────────────────────────────────────────
    # TOP-LEVEL LAYOUT
    # ─────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self, bg=COLORS["bg"], height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="RE", font=FONT_TITLE,
                 bg=COLORS["bg"], fg=COLORS["accent"]).pack(side="left", padx=(28,0), pady=12)
        tk.Label(hdr, text="SYNC", font=FONT_TITLE,
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(side="left", pady=12)
        tk.Label(hdr, text="// EMG ACTIVATION MONITOR",
                 font=FONT_LABEL, bg=COLORS["bg"],
                 fg=COLORS["text3"]).pack(side="left", padx=12, pady=22)

        self.status_dot = tk.Label(hdr, text="●", font=("Courier New", 14),
                                    bg=COLORS["bg"], fg=COLORS["red"])
        self.status_dot.pack(side="right", padx=(0,8), pady=16)
        self.status_lbl = tk.Label(hdr, text="DISCONNECTED",
                                    font=FONT_LABEL, bg=COLORS["bg"], fg=COLORS["text2"])
        self.status_lbl.pack(side="right", pady=16)

        tk.Frame(self, bg=COLORS["border"], height=1).pack(fill="x")

        self.content = tk.Frame(self, bg=COLORS["bg"])
        self.content.pack(fill="both", expand=True)

        self._screens = {}
        self._build_connect_screen()
        self._build_calibrate_screen()
        self._build_session_screen()
        self._build_metrics_screen()
        self._show_screen("CONNECT")

    def _show_screen(self, name):
        for s in self._screens.values():
            s.pack_forget()
        self._screens[name].pack(fill="both", expand=True)
        self.store.screen = name

    # ─────────────────────────────────────────
    # SCREEN: CONNECT
    # ─────────────────────────────────────────

    def _build_connect_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["CONNECT"] = f

        col = tk.Frame(f, bg=COLORS["bg"])
        col.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(col, text="DEVICE SETUP", font=("Courier New", 13, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(pady=(0,4))
        tk.Label(col, text="Connect your ReSync cuff via HC-05 Bluetooth",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text2"], justify="center").pack(pady=(0,20))

        port_row = tk.Frame(col, bg=COLORS["bg"])
        port_row.pack(pady=(0,16))
        tk.Label(port_row, text="COM PORT", font=FONT_LABEL,
                 bg=COLORS["bg"], fg=COLORS["text3"], width=12, anchor="w").pack(side="left")
        self.com_port_var = tk.StringVar(value="COM3")
        tk.Entry(port_row, textvariable=self.com_port_var,
                 font=FONT_MONO, bg=COLORS["surface2"], fg=COLORS["text"],
                 insertbackground=COLORS["text"], relief="flat",
                 highlightthickness=1, highlightbackground=COLORS["border"],
                 width=10).pack(side="left", padx=(8,0))

        self.btn_connect = self._make_button(
            col, "[ CONNECT BLUETOOTH ]",
            command=self._on_connect,
            color=COLORS["accent"], width=28)
        self.btn_connect.pack(pady=8)

        self.connect_msg = tk.Label(col, text="", font=FONT_LABEL,
                                     bg=COLORS["bg"], fg=COLORS["text2"])
        self.connect_msg.pack(pady=4)

        info = tk.Frame(col, bg=COLORS["surface"], highlightthickness=1,
                         highlightbackground=COLORS["border"])
        info.pack(pady=20, fill="x")
        for i, (k, v) in enumerate([
            ("DEVICE",    "HC-05 Bluetooth Module"),
            ("BAUD",      "9600"),
            ("SENSOR",    "MyoWare 2.0 EMG"),
            ("ALGORITHM", "Z-Score / Welford online"),
        ]):
            bg = COLORS["surface"] if i % 2 == 0 else COLORS["surface2"]
            row = tk.Frame(info, bg=bg)
            row.pack(fill="x")
            tk.Label(row, text=k, font=FONT_LABEL, bg=bg, fg=COLORS["text3"],
                     width=12, anchor="w").pack(side="left", padx=12, pady=6)
            tk.Label(row, text=v, font=FONT_LABEL, bg=bg,
                     fg=COLORS["text"], anchor="w").pack(side="left")

    # ─────────────────────────────────────────
    # SCREEN: CALIBRATE
    # ─────────────────────────────────────────

    def _build_calibrate_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["CALIBRATE"] = f

        left = tk.Frame(f, bg=COLORS["bg"], width=380)
        left.pack(side="left", fill="y", padx=(32,0), pady=32)
        left.pack_propagate(False)

        tk.Label(left, text="DEVICE WARMUP", font=("Courier New", 13, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(anchor="w")
        tk.Label(left, text="Building EMG baseline — keep muscle at rest",
                 font=FONT_SMALL, bg=COLORS["bg"], fg=COLORS["text2"]).pack(anchor="w", pady=(2,24))

        self.cal_phase_lbl = tk.Label(left, text="READY TO CALIBRATE",
                                       font=FONT_MED, bg=COLORS["bg"],
                                       fg=COLORS["text"], wraplength=340, justify="left")
        self.cal_phase_lbl.pack(anchor="w", pady=(0,8))

        self.cal_instruction = tk.Label(
            left,
            text="Place the sensor on your muscle, then press\nStart Calibration when ready.",
            font=FONT_SMALL, bg=COLORS["bg"], fg=COLORS["text2"], justify="left")
        self.cal_instruction.pack(anchor="w", pady=(0,16))

        self.btn_start_cal = self._make_button(
            left, "[ START CALIBRATION ]",
            command=self._on_start_cal,
            color=COLORS["accent2"], width=26)
        self.btn_start_cal.pack(anchor="w", pady=(0,20))

        tk.Label(left, text="WARMUP PROGRESS", font=FONT_LABEL,
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(anchor="w")
        prog_bg = tk.Frame(left, bg=COLORS["surface2"], height=8,
                            highlightthickness=1, highlightbackground=COLORS["border"])
        prog_bg.pack(fill="x", pady=(4,20))
        self.cal_prog_fill = tk.Frame(prog_bg, bg=COLORS["accent"], height=8)
        self.cal_prog_fill.place(x=0, y=0, relheight=1, relwidth=0)

        self.cal_status_lbl = tk.Label(left, text="", font=FONT_BIG,
                                        bg=COLORS["bg"], fg=COLORS["accent"])
        self.cal_status_lbl.pack(anchor="w", pady=(0,16))

        self.rep_setup_frame = tk.Frame(left, bg=COLORS["bg"])
        tk.Label(self.rep_setup_frame, text="REP TARGET  (optional)",
                 font=FONT_LABEL, bg=COLORS["bg"], fg=COLORS["text3"]).pack(anchor="w")
        rep_row = tk.Frame(self.rep_setup_frame, bg=COLORS["bg"])
        rep_row.pack(anchor="w", pady=(4,16))
        self.rep_target_var = tk.StringVar(value="")
        tk.Entry(rep_row, textvariable=self.rep_target_var,
                 font=FONT_MONO, bg=COLORS["surface2"], fg=COLORS["text"],
                 insertbackground=COLORS["text"], relief="flat",
                 highlightthickness=1, highlightbackground=COLORS["border"],
                 width=8).pack(side="left")
        tk.Label(rep_row, text="  leave blank = until fatigue",
                 font=FONT_LABEL, bg=COLORS["bg"], fg=COLORS["text3"]).pack(side="left")
        self._make_button(self.rep_setup_frame, "[ START SESSION → ]",
                           command=self._on_start_session,
                           color=COLORS["green"], width=26).pack(anchor="w", pady=4)

        right = tk.Frame(f, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True, padx=24, pady=32)
        self._build_mini_plot(right, "cal")

    # ─────────────────────────────────────────
    # SCREEN: SESSION
    # ─────────────────────────────────────────

    def _build_session_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["SESSION"] = f

        top = tk.Frame(f, bg=COLORS["bg"])
        top.pack(fill="x", padx=24, pady=(20,8))

        # Rep counter card
        rep_card = self._make_card(top, 200, 130)
        rep_card.pack(side="left", padx=(0,12))
        tk.Label(rep_card, text="REPS", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).place(x=16, y=12)
        self.rep_lbl = tk.Label(rep_card, text="0", font=FONT_BIG,
                                 bg=COLORS["surface"], fg=COLORS["green"])
        self.rep_lbl.place(relx=0.5, rely=0.55, anchor="center")

        # Z-score card
        z_card = self._make_card(top, 200, 130)
        z_card.pack(side="left", padx=(0,12))
        tk.Label(z_card, text="Z-SCORE", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).place(x=16, y=12)
        self.z_lbl = tk.Label(z_card, text="0.00", font=FONT_MED,
                               bg=COLORS["surface"], fg=COLORS["accent2"])
        self.z_lbl.place(relx=0.5, rely=0.5, anchor="center")
        self.z_bar_bg = tk.Frame(z_card, bg=COLORS["surface2"], height=6)
        self.z_bar_bg.place(x=12, y=108, width=176)
        self.z_bar_fill = tk.Frame(z_card, bg=COLORS["accent2"], height=6)
        self.z_bar_fill.place(x=12, y=108, width=0)

        # Timer card
        tim_card = self._make_card(top, 200, 130)
        tim_card.pack(side="left", padx=(0,12))
        tk.Label(tim_card, text="ELAPSED", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).place(x=16, y=12)
        self.timer_lbl = tk.Label(tim_card, text="00:00", font=FONT_MED,
                                   bg=COLORS["surface"], fg=COLORS["text"])
        self.timer_lbl.place(relx=0.5, rely=0.5, anchor="center")

        # End session button
        btn_frame = tk.Frame(top, bg=COLORS["bg"])
        btn_frame.pack(side="right")
        self._make_button(btn_frame, "[ END SESSION ]",
                           command=self._on_end_session,
                           color=COLORS["red"], width=18).pack()

        # ── Rep dot progress display ──
        dot_outer = tk.Frame(f, bg=COLORS["bg"])
        dot_outer.pack(fill="x", padx=24, pady=(0,8))

        dot_header = tk.Frame(dot_outer, bg=COLORS["bg"])
        dot_header.pack(fill="x")
        tk.Label(dot_header, text="REP PROGRESS",
                 font=FONT_LABEL, bg=COLORS["bg"],
                 fg=COLORS["text3"]).pack(side="left")
        self.dot_target_lbl = tk.Label(dot_header, text="",
                                        font=FONT_LABEL, bg=COLORS["bg"],
                                        fg=COLORS["text3"])
        self.dot_target_lbl.pack(side="right")

        # Canvas for dots — two rows of 10
        self.dot_canvas = tk.Canvas(dot_outer, bg=COLORS["bg"],
                                     height=60, highlightthickness=0)
        self.dot_canvas.pack(fill="x", pady=(6,0))

        # Live EMG plot
        plot_frame = tk.Frame(f, bg=COLORS["bg"])
        plot_frame.pack(fill="both", expand=True, padx=24, pady=(0,16))
        self._build_main_plot(plot_frame)

    # ─────────────────────────────────────────
    # SCREEN: METRICS
    # ─────────────────────────────────────────

    def _build_metrics_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["METRICS"] = f

        tk.Label(f, text="SESSION COMPLETE", font=("Courier New", 13, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(pady=(16,2))
        tk.Label(f, text="Performance summary", font=FONT_SMALL,
                 bg=COLORS["bg"], fg=COLORS["text2"]).pack(pady=(0,4))

        # New session button + AI summary button at top
        btn_row = tk.Frame(f, bg=COLORS["bg"])
        btn_row.pack(pady=(0,8))
        self._make_button(btn_row, "[ NEW SESSION ]",
                           command=self._on_new_session,
                           color=COLORS["accent"], width=18).pack(side="left", padx=8)
        self._make_button(btn_row, "[ AI PHYSIOTHERAPY SUMMARY ]",
                           command=self._on_ai_summary,
                           color=COLORS["green"], width=28).pack(side="left", padx=8)

        self.stats_row = tk.Frame(f, bg=COLORS["bg"])
        self.stats_row.pack(fill="x", padx=24, pady=(0,10))

        charts = tk.Frame(f, bg=COLORS["bg"])
        charts.pack(fill="both", expand=True, padx=24, pady=(0,8))

        left = tk.Frame(charts, bg=COLORS["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0,8))
        tk.Label(left, text="MUSCLE ACTIVATION  (measured + projected fatigue)",
                 font=FONT_LABEL, bg=COLORS["bg"],
                 fg=COLORS["text3"]).pack(anchor="w", pady=(0,4))
        self._build_peak_z_plot(left)

        right = tk.Frame(charts, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(8,0))
        tk.Label(right, text="FATIGUE ONSET REP  (session history)",
                 font=FONT_LABEL, bg=COLORS["bg"],
                 fg=COLORS["text3"]).pack(anchor="w", pady=(0,4))
        self._build_fatigue_onset_plot(right)
        tk.Label(right, text="AVG REP HOLD TIME  (session history)",
                 font=FONT_LABEL, bg=COLORS["bg"],
                 fg=COLORS["text3"]).pack(anchor="w", pady=(8,4))
        self._build_hold_time_plot(right)

    # ─────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────

    def _build_mini_plot(self, parent, tag):
        fig, ax = plt.subplots(figsize=(5, 2.6), facecolor=COLORS["plot_bg"])
        ax.set_facecolor(COLORS["plot_bg"])
        ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(COLORS["border"])
        ax.set_ylabel("EMG", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        ax.set_xlabel("samples", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        line, = ax.plot([], [], color=COLORS["accent"], linewidth=1.2, alpha=0.9)
        fig.tight_layout(pad=0.8)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, f"mini_ax_{tag}",     ax)
        setattr(self, f"mini_line_{tag}",   line)
        setattr(self, f"mini_canvas_{tag}", canvas)

    def _build_main_plot(self, parent):
        fig, ax = plt.subplots(figsize=(8, 2.6), facecolor=COLORS["plot_bg"])
        ax.set_facecolor(COLORS["plot_bg"])
        ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(COLORS["border"])
        ax.set_ylabel("EMG", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        ax.set_xlabel("samples", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        self.main_line, = ax.plot([], [], color=COLORS["accent2"],
                                   linewidth=1.2, alpha=0.9)
        self.thresh_line = ax.axhline(
            y=0, color=COLORS["yellow"],
            linewidth=1.2, linestyle="--", alpha=0.9,
            label="Contract above here")
        self.deact_line = ax.axhline(
            y=0, color=COLORS["red"],
            linewidth=1.2, linestyle=":", alpha=0.9,
            label="Relax below here")
        ax.legend(facecolor=COLORS["surface"], labelcolor=COLORS["text3"],
                   fontsize=7, loc="upper right")
        fig.tight_layout(pad=0.8)
        self.main_canvas = FigureCanvasTkAgg(fig, master=parent)
        self.main_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.main_ax = ax

    def _build_peak_z_plot(self, parent):
        fig, ax = plt.subplots(figsize=(4.5, 2.8), facecolor=COLORS["plot_bg"])
        ax.set_facecolor(COLORS["plot_bg"])
        ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(COLORS["border"])
        ax.set_xlabel("Rep", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        ax.set_ylabel("Peak Z", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        fig.tight_layout(pad=0.8)
        self.peak_z_canvas = FigureCanvasTkAgg(fig, master=parent)
        self.peak_z_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.peak_z_fig = fig
        self.peak_z_ax  = ax

    def _build_fatigue_onset_plot(self, parent):
        fig, ax = plt.subplots(figsize=(4.5, 1.6), facecolor=COLORS["plot_bg"])
        ax.set_facecolor(COLORS["plot_bg"])
        ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(COLORS["border"])
        ax.set_ylabel("Rep #", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        fig.tight_layout(pad=0.8)
        self.fatigue_onset_canvas = FigureCanvasTkAgg(fig, master=parent)
        self.fatigue_onset_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.fatigue_onset_fig = fig
        self.fatigue_onset_ax  = ax

    def _build_hold_time_plot(self, parent):
        fig, ax = plt.subplots(figsize=(4.5, 1.6), facecolor=COLORS["plot_bg"])
        ax.set_facecolor(COLORS["plot_bg"])
        ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(COLORS["border"])
        ax.set_ylabel("Secs", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        fig.tight_layout(pad=0.8)
        self.hold_time_canvas = FigureCanvasTkAgg(fig, master=parent)
        self.hold_time_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.hold_time_fig = fig
        self.hold_time_ax  = ax

    # ─────────────────────────────────────────
    # REP DOT DISPLAY
    # ─────────────────────────────────────────

    def _draw_dots(self, rep_count, target):
        """Redraw all dots on the canvas based on current rep count."""
        c      = self.dot_canvas
        c.delete("all")
        self._dot_ids = []

        n_dots  = target if target > 0 else max(rep_count, 1)
        n_dots  = min(n_dots, MAX_REPS)

        # Layout — two rows of 10, or single row if ≤10
        cols    = min(n_dots, 10)
        rows    = 2 if n_dots > 10 else 1
        r       = 10    # dot radius
        gap     = 8     # gap between dots
        step    = r * 2 + gap
        total_w = cols * step - gap
        c_w     = c.winfo_width() or 900
        x_start = max(12, (c_w - total_w) // 2)
        y_positions = [20, 48] if rows == 2 else [30]

        for i in range(n_dots):
            row = i // 10
            col = i % 10
            x   = x_start + col * step + r
            y   = y_positions[row] if row < len(y_positions) else y_positions[-1]

            if i < rep_count:
                color = DOT_COLORS[min(i, len(DOT_COLORS)-1)]
            else:
                color = COLORS["dot_empty"]

            oid = c.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline="", tags=f"dot_{i+1}")
            self._dot_ids.append((oid, x, y, i+1))

        # Update target label
        if target > 0:
            self.dot_target_lbl.config(
                text=f"{rep_count} / {target} reps")
        else:
            self.dot_target_lbl.config(
                text=f"{rep_count} reps")

    def _pulse_dot(self, rep_num):
        """Animate a brief glow pulse on dot rep_num then lock it in."""
        c = self.dot_canvas
        if not self._dot_ids:
            return

        # Find the dot
        target_entry = None
        for entry in self._dot_ids:
            oid, x, y, n = entry
            if n == rep_num:
                target_entry = entry
                break

        if target_entry is None:
            return

        oid, x, y, n = target_entry
        final_color   = DOT_COLORS[min(n-1, len(DOT_COLORS)-1)]
        pulse_r       = 15   # enlarged during pulse
        normal_r      = 10

        def step1():
            # Enlarge and flash white
            c.delete(oid)
            new_oid = c.create_oval(
                x - pulse_r, y - pulse_r,
                x + pulse_r, y + pulse_r,
                fill=COLORS["dot_pulse"], outline="",
                tags=f"dot_{n}")
            # Update reference in list
            for idx, entry in enumerate(self._dot_ids):
                if entry[3] == n:
                    self._dot_ids[idx] = (new_oid, x, y, n)
                    break
            self.after(120, lambda: step2(new_oid))

        def step2(pulse_oid):
            # Shrink back to normal size with final color
            c.delete(pulse_oid)
            new_oid = c.create_oval(
                x - normal_r, y - normal_r,
                x + normal_r, y + normal_r,
                fill=final_color, outline="",
                tags=f"dot_{n}")
            for idx, entry in enumerate(self._dot_ids):
                if entry[3] == n:
                    self._dot_ids[idx] = (new_oid, x, y, n)
                    break

        self.after(0, step1)

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _make_button(self, parent, text, command, color, width=20):
        return tk.Button(
            parent, text=text, command=command,
            font=FONT_BTN, fg=color, bg=COLORS["surface"],
            activeforeground=COLORS["bg"], activebackground=color,
            relief="flat", bd=0, cursor="hand2",
            highlightthickness=1, highlightbackground=color,
            width=width, pady=8)

    def _make_card(self, parent, width, height):
        f = tk.Frame(parent, bg=COLORS["surface"], width=width, height=height,
                      highlightthickness=1, highlightbackground=COLORS["border"])
        f.pack_propagate(False)
        return f

    def _stat_card(self, parent, label, value, color=None):
        card = tk.Frame(parent, bg=COLORS["surface"],
                         highlightthickness=1, highlightbackground=COLORS["border"])
        card.pack(side="left", expand=True, fill="both", padx=4)
        tk.Label(card, text=label, font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).pack(pady=(10,2))
        tk.Label(card, text=str(value), font=("Courier New", 18, "bold"),
                 bg=COLORS["surface"],
                 fg=color or COLORS["text"]).pack(pady=(0,10))

    # ─────────────────────────────────────────
    # ACTIONS
    # ─────────────────────────────────────────

    def _on_connect(self):
        if not SERIAL_AVAILABLE:
            self.connect_msg.config(
                text="pyserial not installed — run: pip install pyserial",
                fg=COLORS["red"])
            return
        port = self.com_port_var.get().strip()
        if not port:
            self.connect_msg.config(text="Enter a COM port first.", fg=COLORS["yellow"])
            return
        self.connect_msg.config(text="Connecting...", fg=COLORS["yellow"])
        self.btn_connect.config(state="disabled")

        def attempt():
            try:
                self.reader = BluetoothReader(self.store, port)
                self.reader.start()
                self.store.port_open = True
                self.store.last_rx   = time.time()
                self.after(0, lambda: [
                    self.connect_msg.config(
                        text=f"Connected on {port}", fg=COLORS["green"]),
                    self.after(600, lambda: self._show_screen("CALIBRATE"))
                ])
            except Exception as e:
                self.after(0, lambda err=e: [
                    self.connect_msg.config(text=f"Failed: {err}", fg=COLORS["red"]),
                    self.btn_connect.config(state="normal")
                ])
        threading.Thread(target=attempt, daemon=True).start()

    def _on_start_cal(self):
        if self.reader is None:
            self.cal_phase_lbl.config(text="NOT CONNECTED", fg=COLORS["red"])
            return
        self.reader.send('s')
        self.btn_start_cal.config(state="disabled")
        self.cal_phase_lbl.config(text="COLLECTING BASELINE...", fg=COLORS["accent2"])
        self.cal_instruction.config(
            text="Relax the target muscle completely.\nThe device is sampling your resting signal.")

    def _on_start_session(self):
        target_str = self.rep_target_var.get().strip()
        if target_str.isdigit() and int(target_str) > 0:
            self.store.target_reps = int(target_str)
            self.reader.send(target_str)
        else:
            self.store.target_reps = 0
        self.reader.send('s')
        self.store.session_start    = time.time()
        self.store.rep_durations    = []
        self.store.rep_peaks        = []
        self.store.rep_count        = 0
        self.store.fatigue_window   = []
        self.store.fatigue_onset_rep = None
        self.store.new_rep_pulse    = 0
        self.store.trigger_fatigue_led = False
        self._last_pulsed_rep       = 0
        self._session_ended         = False

        # Draw initial dots
        self.after(100, lambda: self._draw_dots(0, self.store.target_reps))
        self._show_screen("SESSION")

    def _on_end_session(self):
        if self.reader:
            self.reader.send('q')
        self._go_to_metrics()

    def _go_to_metrics(self):
        if self._session_ended:
            return
        self._session_ended = True
        self._compute_metrics()
        self._show_screen("METRICS")

    def _on_ai_summary(self):
        """Generate a physiotherapy-style AI summary using OpenAI."""
        if not OPENAI_AVAILABLE:
            self._show_ai_popup(
                "OpenAI library not installed.\nRun: pip install openai")
            return
        if OPENAI_API_KEY == "your-api-key-here":
            self._show_ai_popup(
                "Please add your OpenAI API key\nto the OPENAI_API_KEY variable.")
            return

        self._show_ai_popup("Generating summary...", loading=True)

        def generate():
            try:
                s      = self.store
                peaks  = s.rep_peaks
                durs   = s.rep_durations
                dur_s  = round(time.time() - s.session_start, 1) \
                         if s.session_start else 0
                avg_hold   = round(float(np.mean(durs)), 2) if durs else 0.0
                onset_rep  = s.fatigue_onset_rep if s.fatigue_onset_rep else "not detected"
                total_reps = s.rep_count
                rep_rate   = round(total_reps / (dur_s / 60), 1) if dur_s > 0 else 0

                # Effort trend description
                trend_msg = "insufficient data"
                if len(peaks) >= 4:
                    arr  = np.array(peaks)
                    mid  = len(arr) // 2
                    m1, _ = np.polyfit(np.arange(1, mid+1), arr[:mid], 1)
                    m2, _ = np.polyfit(np.arange(mid+1, len(arr)+1), arr[mid:], 1)
                    def sd(m):
                        if abs(m) < 0.3: return "stable"
                        return "increasing" if m > 0 else "decreasing"
                    trend_msg = (f"Early session effort was {sd(m1)}, "
                                 f"late session effort was {sd(m2)}")

                prompt = f"""You are a physiotherapist reviewing a patient's quadriceps exercise session. 
Provide a warm, professional summary in 3-4 sentences using plain English. 
Focus on what the data means for their muscle health and recovery. 
Give one specific actionable recommendation for their next session.
Do not use technical jargon like z-score or standard deviation.

Session data:
- Exercise: Quadriceps hold (thigh muscle contractions)
- Total reps completed: {total_reps}
- Session duration: {dur_s} seconds
- Average hold time per rep: {avg_hold} seconds
- Rep rate: {rep_rate} reps per minute
- Fatigue onset: rep {onset_rep}
- Effort trend: {trend_msg}
- Historical average hold time: {round(float(np.mean(HISTORICAL_HOLD_TIME)), 2)} seconds"""

                client   = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.7
                )
                summary = response.choices[0].message.content.strip()
                self.after(0, lambda: self._show_ai_popup(summary))

            except Exception as e:
                self.after(0, lambda err=e:
                    self._show_ai_popup(f"Error generating summary:\n{err}"))

        threading.Thread(target=generate, daemon=True).start()

    def _show_ai_popup(self, text, loading=False):
        """Show or update the AI summary popup window."""
        # Close existing popup if open
        if hasattr(self, '_ai_popup') and self._ai_popup.winfo_exists():
            self._ai_popup.destroy()

        win = tk.Toplevel(self)
        win.title("AI Physiotherapy Summary")
        win.configure(bg=COLORS["bg"])
        win.geometry("560x300")
        win.resizable(True, True)
        self._ai_popup = win

        tk.Label(win, text="AI PHYSIOTHERAPY SUMMARY",
                 font=("Courier New", 11, "bold"),
                 bg=COLORS["bg"], fg=COLORS["green"]).pack(pady=(16,8))

        tk.Frame(win, bg=COLORS["border"], height=1).pack(fill="x", padx=20)

        # Scrollable text area
        txt_frame = tk.Frame(win, bg=COLORS["surface"],
                              highlightthickness=1,
                              highlightbackground=COLORS["border"])
        txt_frame.pack(fill="both", expand=True, padx=20, pady=12)

        txt = tk.Text(txt_frame, font=FONT_SMALL, bg=COLORS["surface"],
                      fg=COLORS["text"] if not loading else COLORS["text2"],
                      wrap="word", relief="flat", padx=12, pady=10,
                      cursor="arrow", state="normal")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", text)
        txt.config(state="disabled")

        self._make_button(win, "[ CLOSE ]",
                           command=win.destroy,
                           color=COLORS["text2"], width=12).pack(pady=(0,12))

    def _on_new_session(self):
        s = self.store
        s.rep_count          = 0
        s.rep_peaks          = []
        s.rep_durations      = []
        s.fatigue_window     = []
        s.fatigue_onset_rep  = None
        s.new_rep_pulse      = 0
        s.session_start      = None
        s.session_complete   = False
        s.trigger_fatigue_led = False
        s.target_reps        = 0
        s.cal_progress       = 1.0
        self._session_ended   = False
        self._last_pulsed_rep = 0
        self.rep_target_var.set("")
        self.cal_phase_lbl.config(text="BASELINE READY", fg=COLORS["green"])
        self.cal_instruction.config(
            text="Set an optional rep target and press Start Session.")
        self.cal_status_lbl.config(text="✓", fg=COLORS["green"])
        self.btn_start_cal.config(state="disabled")
        self.cal_prog_fill.place(relwidth=1.0)
        self.rep_setup_frame.pack(anchor="w", pady=(8,0))
        self._show_screen("CALIBRATE")

    # ─────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────

    def _compute_metrics(self):
        s     = self.store
        peaks = s.rep_peaks
        durs  = s.rep_durations
        dur_s = round(time.time() - s.session_start, 1) if s.session_start else 0
        mm, ss = divmod(int(dur_s), 60)

        total_reps = s.rep_count
        rep_rate   = round(total_reps / (dur_s / 60), 1) if dur_s > 0 else 0
        avg_hold   = round(float(np.mean(durs)), 2) if durs else 0.0
        onset_rep  = s.fatigue_onset_rep if s.fatigue_onset_rep else "N/A"

        # Stat cards
        for w in self.stats_row.winfo_children():
            w.destroy()
        self._stat_card(self.stats_row, "TOTAL REPS",
                         total_reps,       COLORS["green"])
        self._stat_card(self.stats_row, "REP RATE (rpm)",
                         rep_rate,         COLORS["accent2"])
        self._stat_card(self.stats_row, "AVG HOLD TIME",
                         f"{avg_hold}s",   COLORS["accent"])
        self._stat_card(self.stats_row, "FATIGUE ONSET REP",
                         onset_rep,        COLORS["yellow"])

        # ── Muscle activation & fatigue projection plot ──
        self.peak_z_ax.cla()
        self.peak_z_ax.set_facecolor(COLORS["plot_bg"])

        if len(peaks) >= 2:
            xs  = np.arange(1, len(peaks) + 1)
            arr = np.array(peaks, dtype=float)

            # ── Personal baseline from first rep ──
            baseline         = float(arr[0])
            early_thresh     = baseline * 1.40   # 40% above = early fatigue
            late_thresh      = baseline * 0.80   # 20% below = late fatigue

            # ── Smooth fitted curve through real data ──
            deg      = min(2, len(peaks) - 1)
            coeffs   = np.polyfit(xs, arr, deg)
            poly     = np.poly1d(coeffs)
            xs_smooth = np.linspace(1, len(peaks), 200)
            ys_smooth = poly(xs_smooth)

            # ── Simulated projection — inverted U shape ──
            # Uses real data slope to start, peaks above early threshold,
            # then drops below late threshold showing full fatigue cycle
            n_proj    = max(12, len(peaks) + 4)
            t         = np.linspace(0, np.pi, n_proj)
            last_val  = float(poly(len(peaks)))

            # Peak of inverted U sits at ~65% above baseline
            peak_val  = baseline * 1.65
            trough_val = baseline * 0.65   # drops below late threshold

            # Blend from last real value up to peak then down to trough
            proj_vals = last_val + (peak_val - last_val) * np.sin(t) \
                        + (trough_val - last_val) * (1 - np.cos(t)) / 2
            xs_proj   = np.arange(len(peaks) + 1, len(peaks) + n_proj + 1)

            # ── Y axis range ──
            all_y = np.concatenate([arr, proj_vals])
            ymin  = max(0, min(float(arr.min()), trough_val) * 0.75)
            ymax  = max(float(all_y.max()), early_thresh) * 1.25

            # ── Background zones ──
            # Green  = normal (between late and early thresholds)
            # Yellow = early fatigue (above early threshold)
            # Red    = late fatigue  (below late threshold)
            self.peak_z_ax.axhspan(ymin,         late_thresh,  alpha=0.10,
                                    color=COLORS["red"],    zorder=0)
            self.peak_z_ax.axhspan(late_thresh,  early_thresh, alpha=0.08,
                                    color=COLORS["green"],  zorder=0)
            self.peak_z_ax.axhspan(early_thresh, ymax,         alpha=0.10,
                                    color=COLORS["yellow"], zorder=0)

            # ── Two horizontal threshold lines ──
            self.peak_z_ax.axhline(early_thresh,
                                    color=COLORS["yellow"], linewidth=1.4,
                                    linestyle="--", alpha=0.85, zorder=3)
            self.peak_z_ax.axhline(late_thresh,
                                    color=COLORS["red"], linewidth=1.4,
                                    linestyle="--", alpha=0.85, zorder=3)

            # Threshold line labels on left side
            total_x = len(peaks) + n_proj + 1
            self.peak_z_ax.text(
                0.6, early_thresh,
                "  Early fatigue threshold",
                color=COLORS["yellow"], fontsize=6,
                va="bottom", fontfamily="monospace", alpha=0.9)
            self.peak_z_ax.text(
                0.6, late_thresh,
                "  Late fatigue threshold",
                color=COLORS["red"], fontsize=6,
                va="bottom", fontfamily="monospace", alpha=0.9)

            # Zone labels on right
            self.peak_z_ax.text(
                total_x + 0.2, (late_thresh + early_thresh) / 2,
                "Normal\neffort", color=COLORS["green"],
                fontsize=6, va="center", ha="left",
                fontfamily="monospace")
            self.peak_z_ax.text(
                total_x + 0.2, (early_thresh + ymax) / 2,
                "Early\nfatigue", color=COLORS["yellow"],
                fontsize=6, va="center", ha="left",
                fontfamily="monospace")
            self.peak_z_ax.text(
                total_x + 0.2, (ymin + late_thresh) / 2,
                "Late\nfatigue", color=COLORS["red"],
                fontsize=6, va="center", ha="left",
                fontfamily="monospace")

            # ── Real data dots ──
            self.peak_z_ax.scatter(xs, arr,
                                    color=COLORS["accent2"], s=22,
                                    zorder=6, alpha=0.9,
                                    label="Measured activation")

            # ── Smooth curve through real data ──
            self.peak_z_ax.plot(xs_smooth, ys_smooth,
                                 color=COLORS["accent2"],
                                 linewidth=2, alpha=0.9, zorder=5,
                                 label="Activation trend")

            # ── Projected inverted U curve ──
            xs_proj_full = np.concatenate([[len(peaks)], xs_proj])
            ys_proj_full = np.concatenate([[last_val],   proj_vals])
            self.peak_z_ax.plot(xs_proj_full, ys_proj_full,
                                 color=COLORS["yellow"],
                                 linewidth=2, linestyle="--",
                                 alpha=0.85, zorder=5,
                                 label="Projected trajectory")

            # ── Annotate early fatigue crossing on projection ──
            for i in range(1, len(proj_vals)):
                if proj_vals[i-1] < early_thresh <= proj_vals[i]:
                    cross_x = xs_proj[i]
                    self.peak_z_ax.axvline(cross_x, color=COLORS["yellow"],
                                            linewidth=1, linestyle=":",
                                            alpha=0.7, zorder=4)
                    self.peak_z_ax.text(
                        cross_x + 0.2,
                        early_thresh * 1.02,
                        f"Early fatigue\n~rep {int(cross_x)}",
                        color=COLORS["yellow"], fontsize=6,
                        va="bottom", fontfamily="monospace")
                    break

            # ── Annotate late fatigue crossing on projection ──
            crossed_early = False
            for i in range(1, len(proj_vals)):
                if proj_vals[i-1] >= early_thresh:
                    crossed_early = True
                if crossed_early and proj_vals[i] < late_thresh <= proj_vals[i-1]:
                    cross_x = xs_proj[i]
                    self.peak_z_ax.axvline(cross_x, color=COLORS["red"],
                                            linewidth=1, linestyle=":",
                                            alpha=0.7, zorder=4)
                    self.peak_z_ax.text(
                        cross_x + 0.2,
                        late_thresh * 0.97,
                        f"Late fatigue\n~rep {int(cross_x)}",
                        color=COLORS["red"], fontsize=6,
                        va="top", fontfamily="monospace")
                    break

            # ── Real fatigue onset marker if actually detected ──
            onset = s.fatigue_onset_rep
            if isinstance(onset, int):
                self.peak_z_ax.axvline(onset,
                                        color=COLORS["accent"],
                                        linewidth=1.5, linestyle="-.",
                                        alpha=0.9, zorder=4)
                self.peak_z_ax.text(
                    onset + 0.2, ymax * 0.92,
                    f"Fatigue\ndetected\nrep {onset}",
                    color=COLORS["accent"], fontsize=6,
                    va="top", fontfamily="monospace")

            # ── Divider between real and projected ──
            self.peak_z_ax.axvline(len(peaks) + 0.5,
                                    color=COLORS["text3"],
                                    linewidth=0.8, linestyle=":",
                                    alpha=0.4, zorder=3)
            self.peak_z_ax.text(
                len(peaks) + 0.6, ymax * 0.99,
                "← measured  |  projected →",
                color=COLORS["text3"], fontsize=5.5,
                va="top", fontfamily="monospace", alpha=0.7)

            self.peak_z_ax.legend(facecolor=COLORS["surface"],
                                   labelcolor=COLORS["text2"],
                                   fontsize=6, loc="upper left",
                                   framealpha=0.8)
            self.peak_z_ax.set_xlim(0.5, total_x + 2.5)
            self.peak_z_ax.set_ylim(ymin, ymax)

            # x-axis ticks
            real_ticks = list(xs[::max(1, len(xs)//6)])
            proj_ticks = [len(peaks) + n_proj]
            all_ticks  = real_ticks + proj_ticks
            all_labels = [str(t) for t in real_ticks] + \
                         [f"~{len(peaks)+n_proj}"]
            self.peak_z_ax.set_xticks(all_ticks)
            self.peak_z_ax.set_xticklabels(all_labels,
                                            color=COLORS["text3"],
                                            fontsize=6.5,
                                            fontfamily="monospace")

        self.peak_z_ax.set_xlabel("Rep number",
                                   color=COLORS["text3"],
                                   fontsize=7, fontfamily="monospace")
        self.peak_z_ax.set_ylabel("Muscle Activation",
                                   color=COLORS["text3"],
                                   fontsize=7, fontfamily="monospace")
        self.peak_z_ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for sp in self.peak_z_ax.spines.values():
            sp.set_color(COLORS["border"])
        self.peak_z_fig.tight_layout(pad=0.8)
        self.peak_z_canvas.draw_idle()

        # Fatigue onset bar chart
        self.fatigue_onset_ax.cla()
        self.fatigue_onset_ax.set_facecolor(COLORS["plot_bg"])
        hist_onsets = HISTORICAL_FATIGUE_ONSET + \
            ([onset_rep] if isinstance(onset_rep, int) else [0])
        labels = SESSION_LABELS + ["NOW"]
        bar_colors = [COLORS["surface2"]] * 10 + [COLORS["accent"]]
        bars = self.fatigue_onset_ax.bar(labels, hist_onsets,
                                          color=bar_colors, width=0.6, alpha=0.9)
        bars[-1].set_edgecolor(COLORS["accent"])
        bars[-1].set_linewidth(1.5)
        self.fatigue_onset_ax.set_ylabel("Rep #", color=COLORS["text3"],
                                          fontsize=7, fontfamily="monospace")
        self.fatigue_onset_ax.tick_params(colors=COLORS["text3"], labelsize=6,
                                           rotation=30)
        for sp in self.fatigue_onset_ax.spines.values():
            sp.set_color(COLORS["border"])
        self.fatigue_onset_fig.tight_layout(pad=0.8)
        self.fatigue_onset_canvas.draw_idle()

        # Hold time bar chart
        self.hold_time_ax.cla()
        self.hold_time_ax.set_facecolor(COLORS["plot_bg"])
        hist_hold   = HISTORICAL_HOLD_TIME + [avg_hold]
        bar_colors2 = [COLORS["surface2"]] * 10 + [COLORS["yellow"]]
        bars2 = self.hold_time_ax.bar(labels, hist_hold,
                                       color=bar_colors2, width=0.6, alpha=0.9)
        bars2[-1].set_edgecolor(COLORS["yellow"])
        bars2[-1].set_linewidth(1.5)
        hist_mean = np.mean(HISTORICAL_HOLD_TIME)
        self.hold_time_ax.axhline(hist_mean, color=COLORS["text3"],
                                   linewidth=1, linestyle="--", alpha=0.5,
                                   label=f"hist mean {round(hist_mean,2)}s")
        self.hold_time_ax.legend(facecolor=COLORS["surface"],
                                  labelcolor=COLORS["text2"], fontsize=6)
        self.hold_time_ax.set_ylabel("Secs", color=COLORS["text3"],
                                      fontsize=7, fontfamily="monospace")
        self.hold_time_ax.tick_params(colors=COLORS["text3"], labelsize=6,
                                       rotation=30)
        for sp in self.hold_time_ax.spines.values():
            sp.set_color(COLORS["border"])
        self.hold_time_fig.tight_layout(pad=0.8)
        self.hold_time_canvas.draw_idle()

    # ─────────────────────────────────────────
    # UPDATE LOOP  (every 80ms)
    # ─────────────────────────────────────────

    def _on_close(self):
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        if self.reader:
            self.reader.stop()
        self.destroy()

    def _update_loop(self):
        try:
            self._update_loop_body()
        except tk.TclError:
            return

    def _update_loop_body(self):
        s    = self.store
        scr  = s.screen
        hist = list(s.emg_history)

        # Status dot
        if not s.port_open:
            self.status_dot.config(fg=COLORS["red"])
            self.status_lbl.config(text="DISCONNECTED")
        elif time.time() - s.last_rx < 3.0:
            self.status_dot.config(fg=COLORS["green"])
            self.status_lbl.config(text="CONNECTED")
        elif s.arduino_ready:
            self.status_dot.config(fg=COLORS["green"])
            self.status_lbl.config(text="READY")
        elif s.ever_received:
            self.status_dot.config(fg=COLORS["yellow"])
            self.status_lbl.config(text="NO SIGNAL")
        else:
            self.status_dot.config(fg=COLORS["yellow"])
            self.status_lbl.config(text="WAITING FOR ARDUINO")

        # CALIBRATE
        if scr == "CALIBRATE":
            self.cal_prog_fill.place(relwidth=min(1.0, s.cal_progress))
            if s.arduino_ready and not self._cal_ready_shown:
                self._cal_ready_shown = True
                self.cal_phase_lbl.config(text="BASELINE READY", fg=COLORS["green"])
                self.cal_instruction.config(
                    text="Warmup complete. Set an optional rep target\nthen press Start Session.")
                self.cal_status_lbl.config(text="✓", fg=COLORS["green"])
                self.rep_setup_frame.pack(anchor="w", pady=(8,0))
            if len(hist) > 1:
                self.mini_line_cal.set_data(range(len(hist)), hist)
                self.mini_ax_cal.set_xlim(0, len(hist))
                self.mini_ax_cal.set_ylim(max(0, min(hist)-50), max(hist)+50)
                self.mini_canvas_cal.draw_idle()

        # SESSION
        if scr == "SESSION":
            reps   = s.rep_count
            target = s.target_reps

            self.rep_lbl.config(text=str(reps))

            z = s.z_score
            self.z_lbl.config(text=f"{z:+.2f}",
                               fg=COLORS["red"] if z > Z_THRESH else COLORS["accent2"])
            self.z_bar_fill.place(
                width=min(176, max(0, int(abs(z) / Z_SCORE_SCALE * 176))))

            if s.session_start:
                elapsed = int(time.time() - s.session_start)
                mm, ss  = divmod(elapsed, 60)
                self.timer_lbl.config(text=f"{mm:02d}:{ss:02d}")

            # Rep dot updates
            new_pulse = s.new_rep_pulse
            if new_pulse > self._last_pulsed_rep:
                # Redraw dots then pulse the new one
                self._draw_dots(new_pulse - 1, target)  # draw without new dot filled
                self._pulse_dot(new_pulse)               # pulse animates it in
                self._last_pulsed_rep = new_pulse
                # Redraw fully after pulse
                self.after(300, lambda r=new_pulse, t=target:
                            self._draw_dots(r, t))
            elif reps == 0 and self._last_pulsed_rep == 0:
                self._draw_dots(0, target)

            # Live EMG + threshold lines
            if len(hist) > 1:
                self.main_line.set_data(range(len(hist)), hist)
                self.main_ax.set_xlim(0, len(hist))
                self.main_ax.set_ylim(max(0, min(hist)-50), max(hist)+80)
                thresh_emg = s.mu + Z_THRESH * s.sigma
                deact_emg  = s.mu + Z_DEACT_THRESH * s.sigma
                self.thresh_line.set_ydata([thresh_emg, thresh_emg])
                self.deact_line.set_ydata([deact_emg, deact_emg])
                self.main_canvas.draw_idle()

            # Send 'f' to Arduino when Python detects fatigue
            if s.trigger_fatigue_led:
                s.trigger_fatigue_led = False
                if self.reader:
                    self.reader.send('f')

            if s.session_complete:
                self._go_to_metrics()

        self._after_id = self.after(80, self._update_loop)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(0)  # change 2 to 1
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    app = ReSync()
    app.mainloop()
