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

# ─────────────────────────────────────────────
# CONSTANTS  (must match Arduino values)
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
}

FONT_TITLE = ("Courier New", 28, "bold")
FONT_LABEL = ("Courier New", 9)
FONT_BIG   = ("Courier New", 48, "bold")
FONT_MED   = ("Courier New", 20, "bold")
FONT_SMALL = ("Courier New", 10)
FONT_MONO  = ("Courier New", 11)
FONT_BTN   = ("Courier New", 11, "bold")

HISTORY_LEN   = 300
WARMUP_SAMPLES = 200   # matches Arduino WARMUP_SAMPLES
FATIGUE_RATIO  = 1.4   # matches Arduino FATIGUE_RATIO
Z_THRESH       = 2.5   # matches Arduino Z_THRESH

# ─────────────────────────────────────────────
# BLUETOOTH READER
# ─────────────────────────────────────────────

class BluetoothReader:
    """
    Reads serial lines from HC-05 and parses them into Store.
    Arduino serial protocol:
      Warmup:   "Warmup N/200"
      Ready:    "Baseline ready — mean=X sigma=Y"
      Telemetry:"raw=N sm=N μ=N σ=N z=N act=YES|no"
      Rep:      "Rep: N"
      Peak z:   "peakZ=N.NN baseline=N.NN"
      Stop:     "*** FATIGUE DETECTED — rest now ***"
                "*** TARGET REPS REACHED — well done! ***"
                "Session ended early."
    """

    def __init__(self, store, port, baud=9600):
        self.store   = store
        self.port    = port
        self.baud    = baud
        self.ser     = None
        self.running = False

    def start(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=1, write_timeout=2)
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

        # Main telemetry: "raw=512 sm=510 μ=511.0 σ=2.3 z=0.45 act=no"
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
                s.active = (data.get('act') == 'YES')
            except (KeyError, ValueError):
                pass

        # Warmup progress: "Warmup 50/200"
        elif line.startswith('Warmup'):
            try:
                nums = line.split()[1].split('/')
                s.cal_progress = int(nums[0]) / int(nums[1])
            except (IndexError, ValueError, ZeroDivisionError):
                pass

        # Warmup complete — any of these lines signal Arduino is in WAIT_STATE
        elif ('Baseline ready' in line
              or 'run until fatigue' in line
              or 'start a new session' in line):
            s.cal_progress  = 1.0
            s.arduino_ready = True

        # Rep count: "Rep: 5"
        elif line.startswith('Rep:'):
            try:
                s.rep_count = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass

        # Peak z after rep: "peakZ=3.20 baseline=2.50"
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
                if base > 0 and peak > base:
                    # Scale 0→0%, ratio>=FATIGUE_RATIO→100%
                    s.fatigue = min(1.0, (peak / base - 1.0) / (FATIGUE_RATIO - 1.0))
            except (KeyError, ValueError):
                pass

        # Session stop conditions — Arduino resets itself to WAIT_STATE,
        # GUI navigates to METRICS then back to CALIBRATE for next session.
        elif 'FATIGUE DETECTED' in line:
            s.fatigue          = 1.0
            s.session_complete = True

        elif 'TARGET REPS REACHED' in line:
            s.session_complete = True

        elif 'Session ended early' in line:
            s.session_complete = True

        # Arduino back in WAIT_STATE after a session — signal GUI to reset
        elif 'start a new session' in line:
            s.arduino_in_wait = True

        # Mark data as flowing and record timestamp for signal health check
        s.last_rx       = time.time()
        s.ever_received = True


# ─────────────────────────────────────────────
# APP STATE
# ─────────────────────────────────────────────

class Store:
    def __init__(self):
        self.screen           = "CONNECT"
        self.connected        = False
        self.emg_raw          = 0
        self.emg_history      = []
        self.z_score          = 0.0
        self.active           = False
        self.rep_count        = 0
        self.rep_peaks        = []
        self.fatigue          = 0.0
        self.cal_progress     = 0.0
        self.session_start    = None
        self.metrics          = {}
        self.arduino_ready    = False   # warmup complete, can start session
        self.arduino_in_wait  = False   # Arduino returned to WAIT_STATE after session
        self.session_complete = False   # fatigue / reps reached / quit
        self.target_reps      = 0       # 0 = no limit
        self.port_open        = False   # COM port opened successfully
        self.last_rx          = 0.0     # timestamp of last received serial line
        self.ever_received    = False   # True once any data has arrived


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

class ReSync(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ReSync")
        self.geometry("1000x680")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        self.store  = Store()
        self.reader = None

        self._cal_ready_shown = False
        self._session_ended   = False
        self._after_id        = None

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
                 bg=COLORS["bg"], fg=COLORS["accent"]).pack(side="left", padx=(28, 0), pady=12)
        tk.Label(hdr, text="SYNC", font=FONT_TITLE,
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(side="left", pady=12)
        tk.Label(hdr, text="// EMG ACTIVATION MONITOR",
                 font=FONT_LABEL, bg=COLORS["bg"],
                 fg=COLORS["text3"]).pack(side="left", padx=12, pady=22)

        self.status_dot = tk.Label(hdr, text="●", font=("Courier New", 14),
                                    bg=COLORS["bg"], fg=COLORS["red"])
        self.status_dot.pack(side="right", padx=(0, 8), pady=16)
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
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(pady=(0, 4))
        tk.Label(col, text="Connect your ReSync cuff via HC-05 Bluetooth",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text2"], justify="center").pack(pady=(0, 20))

        # COM port entry
        port_row = tk.Frame(col, bg=COLORS["bg"])
        port_row.pack(pady=(0, 16))
        tk.Label(port_row, text="COM PORT", font=FONT_LABEL,
                 bg=COLORS["bg"], fg=COLORS["text3"], width=12, anchor="w").pack(side="left")
        self.com_port_var = tk.StringVar(value="COM3")
        tk.Entry(port_row, textvariable=self.com_port_var,
                 font=FONT_MONO, bg=COLORS["surface2"], fg=COLORS["text"],
                 insertbackground=COLORS["text"], relief="flat",
                 highlightthickness=1, highlightbackground=COLORS["border"],
                 width=10).pack(side="left", padx=(8, 0))

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
        rows = [
            ("DEVICE",    "HC-05 Bluetooth Module"),
            ("BAUD",      "9600"),
            ("SENSOR",    "MyoWare 2.0 EMG"),
            ("ALGORITHM", "Z-Score / Welford online"),
        ]
        for i, (k, v) in enumerate(rows):
            bg = COLORS["surface"] if i % 2 == 0 else COLORS["surface2"]
            row = tk.Frame(info, bg=bg)
            row.pack(fill="x")
            tk.Label(row, text=k, font=FONT_LABEL, bg=bg, fg=COLORS["text3"],
                     width=12, anchor="w").pack(side="left", padx=12, pady=6)
            tk.Label(row, text=v, font=FONT_LABEL, bg=bg,
                     fg=COLORS["text"], anchor="w").pack(side="left")

    # ─────────────────────────────────────────
    # SCREEN: CALIBRATE  (Arduino warmup + session setup)
    # ─────────────────────────────────────────

    def _build_calibrate_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["CALIBRATE"] = f

        left = tk.Frame(f, bg=COLORS["bg"], width=380)
        left.pack(side="left", fill="y", padx=(32, 0), pady=32)
        left.pack_propagate(False)

        tk.Label(left, text="DEVICE WARMUP", font=("Courier New", 13, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(anchor="w")
        tk.Label(left, text="Building EMG baseline — keep muscle at rest",
                 font=FONT_SMALL, bg=COLORS["bg"], fg=COLORS["text2"]).pack(anchor="w", pady=(2, 24))

        self.cal_phase_lbl = tk.Label(left, text="READY TO CALIBRATE",
                                       font=FONT_MED, bg=COLORS["bg"],
                                       fg=COLORS["text"], wraplength=340, justify="left")
        self.cal_phase_lbl.pack(anchor="w", pady=(0, 8))

        self.cal_instruction = tk.Label(
            left,
            text="Place the sensor on your muscle, then press\nStart Calibration when ready.",
            font=FONT_SMALL, bg=COLORS["bg"], fg=COLORS["text2"], justify="left")
        self.cal_instruction.pack(anchor="w", pady=(0, 16))

        self.btn_start_cal = self._make_button(
            left, "[ START CALIBRATION ]",
            command=self._on_start_cal,
            color=COLORS["accent2"], width=26)
        self.btn_start_cal.pack(anchor="w", pady=(0, 20))

        # Progress bar
        self.cal_prog_frame = tk.Frame(left, bg=COLORS["bg"])
        self.cal_prog_frame.pack(anchor="w", fill="x", pady=(0, 4))
        tk.Label(self.cal_prog_frame, text="WARMUP PROGRESS", font=FONT_LABEL,
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(anchor="w")
        prog_bg = tk.Frame(self.cal_prog_frame, bg=COLORS["surface2"], height=8,
                            highlightthickness=1, highlightbackground=COLORS["border"])
        prog_bg.pack(fill="x", pady=(4, 20))
        self.cal_prog_fill = tk.Frame(prog_bg, bg=COLORS["accent"], height=8)
        self.cal_prog_fill.place(x=0, y=0, relheight=1, relwidth=0)

        self.cal_status_lbl = tk.Label(left, text="", font=FONT_BIG,
                                        bg=COLORS["bg"], fg=COLORS["accent"])
        self.cal_status_lbl.pack(anchor="w", pady=(0, 16))

        # Rep target + start button — shown only after warmup completes
        self.rep_setup_frame = tk.Frame(left, bg=COLORS["bg"])

        tk.Label(self.rep_setup_frame, text="REP TARGET  (optional)",
                 font=FONT_LABEL, bg=COLORS["bg"], fg=COLORS["text3"]).pack(anchor="w")
        rep_row = tk.Frame(self.rep_setup_frame, bg=COLORS["bg"])
        rep_row.pack(anchor="w", pady=(4, 16))
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

        # Right: live signal plot during warmup
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
        top.pack(fill="x", padx=24, pady=(20, 8))

        rep_card = self._make_card(top, 200, 130)
        rep_card.pack(side="left", padx=(0, 12))
        tk.Label(rep_card, text="REPS", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).place(x=16, y=12)
        self.rep_lbl = tk.Label(rep_card, text="0", font=FONT_BIG,
                                 bg=COLORS["surface"], fg=COLORS["green"])
        self.rep_lbl.place(relx=0.5, rely=0.55, anchor="center")

        z_card = self._make_card(top, 200, 130)
        z_card.pack(side="left", padx=(0, 12))
        tk.Label(z_card, text="Z-SCORE", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).place(x=16, y=12)
        self.z_lbl = tk.Label(z_card, text="0.00", font=FONT_MED,
                               bg=COLORS["surface"], fg=COLORS["accent2"])
        self.z_lbl.place(relx=0.5, rely=0.5, anchor="center")
        self.z_bar_bg = tk.Frame(z_card, bg=COLORS["surface2"], height=6)
        self.z_bar_bg.place(x=12, y=108, width=176)
        self.z_bar_fill = tk.Frame(z_card, bg=COLORS["accent2"], height=6)
        self.z_bar_fill.place(x=12, y=108, width=0)

        tim_card = self._make_card(top, 200, 130)
        tim_card.pack(side="left", padx=(0, 12))
        tk.Label(tim_card, text="ELAPSED", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).place(x=16, y=12)
        self.timer_lbl = tk.Label(tim_card, text="00:00", font=FONT_MED,
                                   bg=COLORS["surface"], fg=COLORS["text"])
        self.timer_lbl.place(relx=0.5, rely=0.5, anchor="center")

        btn_frame = tk.Frame(top, bg=COLORS["bg"])
        btn_frame.pack(side="right")
        self._make_button(btn_frame, "[ END SESSION ]",
                           command=self._on_end_session,
                           color=COLORS["red"], width=18).pack()

        fat_frame = tk.Frame(f, bg=COLORS["bg"])
        fat_frame.pack(fill="x", padx=24, pady=(0, 8))
        tk.Label(fat_frame, text="FATIGUE INDEX", font=FONT_LABEL,
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(side="left", padx=(0, 12))
        fat_bg = tk.Frame(fat_frame, bg=COLORS["surface2"], height=14,
                           highlightthickness=1, highlightbackground=COLORS["border"])
        fat_bg.pack(side="left", fill="x", expand=True)
        self.fat_fill = tk.Frame(fat_bg, bg=COLORS["green"], height=14)
        self.fat_fill.place(x=0, y=0, relheight=1, relwidth=0)
        self.fat_lbl = tk.Label(fat_frame, text="0%", font=FONT_LABEL,
                                 bg=COLORS["bg"], fg=COLORS["text2"], width=5)
        self.fat_lbl.pack(side="left", padx=(8, 0))

        plot_frame = tk.Frame(f, bg=COLORS["bg"])
        plot_frame.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        self._build_main_plot(plot_frame)

    # ─────────────────────────────────────────
    # SCREEN: METRICS
    # ─────────────────────────────────────────

    def _build_metrics_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["METRICS"] = f

        tk.Label(f, text="SESSION COMPLETE", font=("Courier New", 13, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(pady=(24, 2))
        tk.Label(f, text="Performance summary", font=FONT_SMALL,
                 bg=COLORS["bg"], fg=COLORS["text2"]).pack(pady=(0, 16))

        cols = tk.Frame(f, bg=COLORS["bg"])
        cols.pack(fill="both", expand=True, padx=24)

        left = tk.Frame(cols, bg=COLORS["bg"], width=320)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.pack_propagate(False)
        self.metrics_frame = tk.Frame(left, bg=COLORS["bg"])
        self.metrics_frame.pack(fill="both", expand=True)

        right = tk.Frame(cols, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True)
        self._build_metrics_plot(right)

        btn_row = tk.Frame(f, bg=COLORS["bg"])
        btn_row.pack(pady=16)
        self._make_button(btn_row, "[ NEW SESSION ]",
                           command=self._on_new_session,
                           color=COLORS["accent"], width=18).pack(side="left", padx=8)

    # ─────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────

    def _build_mini_plot(self, parent, tag):
        fig, ax = plt.subplots(figsize=(5, 2.6), facecolor=COLORS["plot_bg"])
        ax.set_facecolor(COLORS["plot_bg"])
        ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
        ax.set_ylabel("EMG", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        ax.set_xlabel("samples", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        line, = ax.plot([], [], color=COLORS["accent"], linewidth=1.2, alpha=0.9)
        fig.tight_layout(pad=0.8)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, f"mini_ax_{tag}", ax)
        setattr(self, f"mini_line_{tag}", line)
        setattr(self, f"mini_canvas_{tag}", canvas)

    def _build_main_plot(self, parent):
        fig, ax = plt.subplots(figsize=(8, 2.8), facecolor=COLORS["plot_bg"])
        ax.set_facecolor(COLORS["plot_bg"])
        ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
        ax.set_ylabel("EMG", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        ax.set_xlabel("samples", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        self.main_line, = ax.plot([], [], color=COLORS["accent2"], linewidth=1.2, alpha=0.9)
        # Threshold line at Z_THRESH — shown as a horizontal reference
        self.thresh_line = ax.axhline(y=0, color=COLORS["yellow"], linewidth=1,
                                       linestyle="--", alpha=0.6, visible=False)
        fig.tight_layout(pad=0.8)
        self.main_canvas = FigureCanvasTkAgg(fig, master=parent)
        self.main_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.main_ax = ax

    def _build_metrics_plot(self, parent):
        self.met_fig, self.met_ax = plt.subplots(figsize=(5, 3.2), facecolor=COLORS["plot_bg"])
        self.met_ax.set_facecolor(COLORS["plot_bg"])
        self.met_ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for spine in self.met_ax.spines.values():
            spine.set_color(COLORS["border"])
        self.met_canvas = FigureCanvasTkAgg(self.met_fig, master=parent)
        self.met_canvas.get_tk_widget().pack(fill="both", expand=True)

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

    def _stat_row(self, parent, label, value, color=None):
        row = tk.Frame(parent, bg=COLORS["surface"],
                        highlightthickness=1, highlightbackground=COLORS["border"])
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, font=FONT_LABEL, bg=COLORS["surface"],
                 fg=COLORS["text3"], width=22, anchor="w").pack(side="left", padx=12, pady=8)
        tk.Label(row, text=str(value), font=FONT_LABEL, bg=COLORS["surface"],
                 fg=color or COLORS["text"]).pack(side="right", padx=12)

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
                self.store.last_rx   = time.time()  # grace period — Arduino may be silent
                self.after(0, lambda: [
                    self.connect_msg.config(text=f"Connected on {port}", fg=COLORS["green"]),
                    self.after(600, lambda: self._show_screen("CALIBRATE"))
                ])
            except Exception as e:
                self.after(0, lambda err=e: [
                    self.connect_msg.config(text=f"Failed: {err}", fg=COLORS["red"]),
                    self.btn_connect.config(state="normal")
                ])

        threading.Thread(target=attempt, daemon=True).start()

    def _on_start_cal(self):
        """Send 's' to Arduino to begin warmup, then show progress bar."""
        if self.reader is None:
            self.cal_phase_lbl.config(text="NOT CONNECTED", fg=COLORS["red"])
            return
        try:
            self.reader.send('s')
        except Exception as e:
            self.cal_phase_lbl.config(text=f"Send failed: {e}", fg=COLORS["red"])
            return
        self.btn_start_cal.config(state="disabled")
        self.cal_phase_lbl.config(text="COLLECTING BASELINE...", fg=COLORS["accent2"])
        self.cal_instruction.config(
            text="Relax the target muscle completely.\nThe device is sampling your resting signal.")

    def _on_start_session(self):
        """Send optional rep target then 's' to Arduino, then show SESSION screen."""
        target_str = self.rep_target_var.get().strip()
        if target_str.isdigit() and int(target_str) > 0:
            self.store.target_reps = int(target_str)
            self.reader.send(target_str)   # Arduino receives number, sets targetReps
        else:
            self.store.target_reps = 0
        self.reader.send('s')              # Arduino transitions WAIT_STATE → RUNNING_STATE
        self.store.session_start = time.time()
        self._session_ended = False
        self._show_screen("SESSION")

    def _on_end_session(self):
        """User pressed End Session — send 'q' to Arduino then show metrics."""
        if self.reader:
            self.reader.send('q')          # Arduino calls endSession(), resets to WAIT_STATE
        self._go_to_metrics()

    def _go_to_metrics(self):
        if self._session_ended:
            return
        self._session_ended = True
        self._compute_metrics()
        self._show_screen("METRICS")

    def _on_new_session(self):
        """
        Arduino already reset itself to WAIT_STATE after the session ended.
        GUI resets its own counters and goes back to CALIBRATE/setup screen
        so the user can set a rep target and press Start again.
        Warmup is NOT re-run — Arduino kept the Welford baseline.
        """
        s = self.store
        s.rep_count        = 0
        s.rep_peaks        = []
        s.fatigue          = 0.0
        s.session_start    = None
        s.session_complete = False
        s.arduino_in_wait  = False
        s.target_reps      = 0
        s.cal_progress     = 1.0          # warmup already done
        self._session_ended   = False

        self.rep_target_var.set("")
        self.cal_phase_lbl.config(text="BASELINE READY", fg=COLORS["green"])
        self.cal_instruction.config(
            text="Set an optional rep target and press Start Session.")
        self.cal_status_lbl.config(text="✓", fg=COLORS["green"])
        self.btn_start_cal.config(state="disabled")  # warmup already done, no re-cal needed
        self.cal_prog_fill.place(relwidth=1.0)
        self.rep_setup_frame.pack(anchor="w", pady=(8, 0))
        self._show_screen("CALIBRATE")

    def _compute_metrics(self):
        s     = self.store
        peaks = s.rep_peaks if s.rep_peaks else [0]
        dur   = round(time.time() - s.session_start, 1) if s.session_start else 0

        s.metrics = {
            "Total reps":           s.rep_count,
            "Session duration":     f"{dur}s",
            "Avg peak z-score":     round(float(np.mean(peaks)), 2),
            "Peak consistency (σ)": round(float(np.std(peaks)), 2),
            "Fatigue index":        f"{round(s.fatigue * 100)}%",
        }

        for w in self.metrics_frame.winfo_children():
            w.destroy()

        colors_map = {
            "Total reps":           COLORS["green"],
            "Session duration":     COLORS["text"],
            "Avg peak z-score":     COLORS["accent2"],
            "Peak consistency (σ)": COLORS["text"],
            "Fatigue index":        COLORS["yellow"],
        }
        for k, v in s.metrics.items():
            self._stat_row(self.metrics_frame, k, v, colors_map.get(k))

        self.met_ax.cla()
        self.met_ax.set_facecolor(COLORS["plot_bg"])
        if len(peaks) > 1:
            bars = self.met_ax.bar(range(1, len(peaks) + 1), peaks,
                                    color=COLORS["accent"], width=0.6, alpha=0.85)
            for i, bar in enumerate(bars):
                fade = i / max(len(bars) - 1, 1)
                bar.set_facecolor(plt.cm.RdYlGn(1 - fade * 0.8))
        self.met_ax.set_xlabel("Rep", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        self.met_ax.set_ylabel("Peak z-score", color=COLORS["text3"], fontsize=7, fontfamily="monospace")
        self.met_ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for spine in self.met_ax.spines.values():
            spine.set_color(COLORS["border"])
        self.met_fig.tight_layout(pad=0.8)
        self.met_canvas.draw_idle()

    # ─────────────────────────────────────────
    # UPDATE LOOP  (every 80 ms)
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
            return  # window was closed, stop scheduling

    def _update_loop_body(self):
        s    = self.store
        scr  = s.screen
        hist = list(s.emg_history)

        # Header status dot — reflects actual data flow, not just port open
        if not s.port_open:
            self.status_dot.config(fg=COLORS["red"])
            self.status_lbl.config(text="DISCONNECTED")
        elif time.time() - s.last_rx < 3.0:
            self.status_dot.config(fg=COLORS["green"])
            self.status_lbl.config(text="CONNECTED")
        elif s.arduino_ready and scr in ("CALIBRATE", "SESSION"):
            # Arduino in WAIT_STATE / RUNNING_STATE — silence is expected
            self.status_dot.config(fg=COLORS["green"])
            self.status_lbl.config(text="READY")
        elif s.ever_received:
            self.status_dot.config(fg=COLORS["yellow"])
            self.status_lbl.config(text="NO SIGNAL")
        else:
            self.status_dot.config(fg=COLORS["yellow"])
            self.status_lbl.config(text="WAITING FOR ARDUINO")

        # ── Warmup / setup screen ───────────────
        if scr == "CALIBRATE":
            self.cal_prog_fill.place(relwidth=min(1.0, s.cal_progress))

            # Stale Arduino: raw= data is flowing but warmup never happened this session
            if not s.arduino_ready and not self._cal_ready_shown and len(hist) > 20 and s.cal_progress == 0.0:
                self.cal_phase_lbl.config(text="DEVICE ALREADY RUNNING", fg=COLORS["yellow"])
                self.cal_instruction.config(
                    text="Arduino is in an active session from before.\nPower-cycle it (unplug/replug), then click Start Calibration.")

            # Reveal rep-target entry and Start button once warmup is done
            if s.arduino_ready and not self._cal_ready_shown:
                self._cal_ready_shown = True
                self.cal_phase_lbl.config(text="BASELINE READY", fg=COLORS["green"])
                self.cal_instruction.config(
                    text="Warmup complete. Set an optional rep target\nthen press Start Session.")
                self.cal_status_lbl.config(text="✓", fg=COLORS["green"])
                self.rep_setup_frame.pack(anchor="w", pady=(8, 0))

            if len(hist) > 1:
                self.mini_line_cal.set_data(range(len(hist)), hist)
                self.mini_ax_cal.set_xlim(0, len(hist))
                self.mini_ax_cal.set_ylim(max(0, min(hist) - 50), max(hist) + 50)
                self.mini_canvas_cal.draw_idle()

        # ── Session screen ──────────────────────
        if scr == "SESSION":
            self.rep_lbl.config(text=str(s.rep_count))

            z = s.z_score
            self.z_lbl.config(text=f"{z:+.2f}",
                               fg=COLORS["red"] if z > Z_THRESH else COLORS["accent2"])
            self.z_bar_fill.place(width=min(176, max(0, int(abs(z) / 6.0 * 176))))

            if s.session_start:
                elapsed = int(time.time() - s.session_start)
                mm, ss  = divmod(elapsed, 60)
                self.timer_lbl.config(text=f"{mm:02d}:{ss:02d}")

            pct   = s.fatigue
            color = COLORS["green"] if pct < 0.5 else \
                    COLORS["yellow"] if pct < 0.8 else COLORS["red"]
            self.fat_fill.place(relwidth=pct)
            self.fat_fill.config(bg=color)
            self.fat_lbl.config(text=f"{int(pct * 100)}%")

            if len(hist) > 1:
                self.main_line.set_data(range(len(hist)), hist)
                self.main_ax.set_xlim(0, len(hist))
                self.main_ax.set_ylim(max(0, min(hist) - 50), max(hist) + 80)
                self.main_canvas.draw_idle()

            # Auto-navigate to metrics when Arduino signals session done
            if s.session_complete:
                self._go_to_metrics()

        self._after_id = self.after(80, self._update_loop)



# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = ReSync()
    app.mainloop()
