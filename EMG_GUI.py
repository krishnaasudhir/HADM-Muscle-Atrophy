"""
ReSync — EMG Thigh Activation Monitor
======================================
Real hardware version — reads from Arduino over serial (COM3).

Install dependencies:
    pip install matplotlib numpy pyserial

Run:
    python resync_app.py
"""

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
import re
import serial

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SERIAL_PORT  = "COM3"   # change if your port is different
BAUD_RATE    = 9600
HISTORY_LEN  = 300
WARMUP_TOTAL = 200      # must match Arduino WARMUP_SAMPLES

# ─────────────────────────────────────────────
# COLORS & FONTS
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
FONT_BTN   = ("Courier New", 11, "bold")

# ─────────────────────────────────────────────
# APP STATE
# ─────────────────────────────────────────────

class Store:
    def __init__(self):
        self.screen         = "CONNECT"
        self.connected      = False
        self.emg_raw        = 0
        self.emg_history    = []
        self.z_score        = 0.0
        self.mu             = 0.0
        self.sigma          = 1.0
        self.act            = False
        self.rep_count      = 0
        self.rep_peaks      = []
        self.fatigue        = 0.0
        self.fatigued       = False
        self.warmup_count   = 0
        self.warmup_done    = False
        self.baseline_mean  = None
        self.baseline_sigma = None
        self.session_start  = None
        self.session_active = False
        self.metrics        = {}
        self._lock          = threading.Lock()

store = Store()

# ─────────────────────────────────────────────
# SERIAL READER  (background thread)
# ─────────────────────────────────────────────

class SerialReader:
    def __init__(self, store):
        self.store   = store
        self.ser     = None
        self.running = False

    def connect(self, on_success, on_fail):
        def _run():
            try:
                self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
                time.sleep(2)
                self.ser.reset_input_buffer()
                self.running = True
                with self.store._lock:
                    self.store.connected = True
                on_success()
                self._read_loop()
            except Exception as e:
                on_fail(str(e))
        threading.Thread(target=_run, daemon=True).start()

    def send(self, msg):
        """Send a command string to the Arduino."""
        if self.ser and self.ser.is_open:
            self.ser.write((msg + "\n").encode("utf-8"))

    def _read_loop(self):
        while self.running:
            try:
                if self.ser.in_waiting:
                    raw  = self.ser.readline()
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if line:
                        self._parse(line)
            except Exception:
                break
            time.sleep(0.001)

    def _parse(self, line):
        s = self.store

        # ── Warmup progress: "Warmup 45/200" ──
        m = re.match(r"Warmup\s+(\d+)/(\d+)", line)
        if m:
            with s._lock:
                s.warmup_count = int(m.group(1))
            return

        # ── Baseline ready ──
        m = re.search(r"Baseline ready.*mean=([\d.]+).*sigma=([\d.]+)", line)
        if m:
            with s._lock:
                s.baseline_mean  = float(m.group(1))
                s.baseline_sigma = float(m.group(2))
                s.warmup_done    = True
                s.warmup_count   = WARMUP_TOTAL
            return

        # ── Rep detected: "Rep: 5" ──
        m = re.match(r"Rep:\s*(\d+)", line)
        if m:
            with s._lock:
                s.rep_count = int(m.group(1))
            return

        # ── Fatigue detected ──
        if "FATIGUE DETECTED" in line:
            with s._lock:
                s.fatigued       = True
                s.fatigue        = 1.0
                s.session_active = False
            return

        # ── Target reps reached ──
        if "TARGET REPS REACHED" in line:
            with s._lock:
                s.session_active = False
            return

        # ── Main data stream ──
        # "raw=523 sm=518 μ=512.3 σ=14.2 z=2.81 act=YES"
        m = re.search(
            r"raw=(\d+).*sm=(\d+).*[=]([\d.]+).*[=]([\d.]+).*z=([-\d.]+).*act=(\w+)",
            line)
        if m:
            raw = int(m.group(1))
            sm  = int(m.group(2))
            mu  = float(m.group(3))
            sig = float(m.group(4))
            z   = float(m.group(5))
            act = m.group(6).upper() == "YES"
            with s._lock:
                s.emg_raw  = raw
                s.z_score  = z
                s.mu       = mu
                s.sigma    = sig
                s.act      = act
                s.emg_history.append(sm)
                if len(s.emg_history) > HISTORY_LEN:
                    s.emg_history.pop(0)
            return

        # ── Peak z per rep (falling edge) ──
        m = re.search(r"peakZ=([\d.]+)", line)
        if m:
            with s._lock:
                s.rep_peaks.append(float(m.group(1)))
                # Update fatigue index from rep peak trend
                peaks = s.rep_peaks
                if len(peaks) >= 4:
                    baseline = float(np.mean(peaks[:3]))
                    recent   = float(peaks[-1])
                    if baseline > 0:
                        s.fatigue = min(1.0, max(0.0,
                            (recent - baseline) / baseline))
            return

serial_reader = SerialReader(store)

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
        self._build_ui()
        self._update_loop()

    # ── Top-level layout ──────────────────────

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
                                    font=FONT_LABEL, bg=COLORS["bg"],
                                    fg=COLORS["text2"])
        self.status_lbl.pack(side="right", pady=16)

        tk.Frame(self, bg=COLORS["border"], height=1).pack(fill="x")

        self.content = tk.Frame(self, bg=COLORS["bg"])
        self.content.pack(fill="both", expand=True)

        self._screens = {}
        self._build_connect_screen()
        self._build_warmup_screen()
        self._build_calibrate_screen()
        self._build_session_screen()
        self._build_metrics_screen()
        self._show_screen("CONNECT")

    def _show_screen(self, name):
        for s in self._screens.values():
            s.pack_forget()
        self._screens[name].pack(fill="both", expand=True)
        store.screen = name

    # ── SCREEN: CONNECT ───────────────────────

    def _build_connect_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["CONNECT"] = f

        col = tk.Frame(f, bg=COLORS["bg"])
        col.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(col, text="DEVICE SETUP",
                 font=("Courier New", 13, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(pady=(0,4))
        tk.Label(col, text="Connect your ReSync cuff to begin",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text2"]).pack(pady=(0,24))

        self.btn_connect = self._make_button(
            col, f"[ CONNECT  {SERIAL_PORT} ]",
            command=self._on_connect,
            color=COLORS["accent"], width=26)
        self.btn_connect.pack(pady=8)

        self.connect_msg = tk.Label(col, text="",
                                     font=FONT_LABEL, bg=COLORS["bg"],
                                     fg=COLORS["text2"])
        self.connect_msg.pack(pady=4)

        info = tk.Frame(col, bg=COLORS["surface"],
                         highlightthickness=1,
                         highlightbackground=COLORS["border"])
        info.pack(pady=20, fill="x")
        for i, (k, v) in enumerate([
            ("PORT",      SERIAL_PORT),
            ("BAUD",      str(BAUD_RATE)),
            ("MODULE",    "HC-05 Bluetooth"),
            ("SENSOR",    "MyoWare 2.0 EMG"),
            ("PLACEMENT", "Rectus Femoris"),
        ]):
            bg = COLORS["surface"] if i % 2 == 0 else COLORS["surface2"]
            row = tk.Frame(info, bg=bg)
            row.pack(fill="x")
            tk.Label(row, text=k, font=FONT_LABEL, bg=bg,
                     fg=COLORS["text3"], width=12, anchor="w"
                     ).pack(side="left", padx=12, pady=6)
            tk.Label(row, text=v, font=FONT_LABEL, bg=bg,
                     fg=COLORS["text"], anchor="w").pack(side="left")

    # ── SCREEN: WARMUP ────────────────────────

    def _build_warmup_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["WARMUP"] = f

        col = tk.Frame(f, bg=COLORS["bg"])
        col.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(col, text="BUILDING BASELINE",
                 font=("Courier New", 13, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(pady=(0,4))
        tk.Label(col,
                 text="Relax your thigh completely.\nDo not move until complete.",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text2"], justify="center").pack(pady=(0,24))

        self.warmup_count_lbl = tk.Label(col, text=f"0 / {WARMUP_TOTAL}",
                                          font=FONT_MED,
                                          bg=COLORS["bg"], fg=COLORS["accent"])
        self.warmup_count_lbl.pack(pady=(0,12))

        prog_bg = tk.Frame(col, bg=COLORS["surface2"], height=10,
                            highlightthickness=1,
                            highlightbackground=COLORS["border"], width=340)
        prog_bg.pack(pady=(0,20))
        prog_bg.pack_propagate(False)
        self.warmup_fill = tk.Frame(prog_bg, bg=COLORS["accent"], height=10)
        self.warmup_fill.place(x=0, y=0, relheight=1, relwidth=0)

        tk.Label(col, text="Baseline will be ready automatically...",
                 font=FONT_LABEL, bg=COLORS["bg"],
                 fg=COLORS["text3"]).pack()

    # ── SCREEN: CALIBRATE ─────────────────────

    def _build_calibrate_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["CALIBRATE"] = f

        left = tk.Frame(f, bg=COLORS["bg"], width=400)
        left.pack(side="left", fill="y", padx=(32,0), pady=32)
        left.pack_propagate(False)

        tk.Label(left, text="READY",
                 font=("Courier New", 13, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(anchor="w")
        tk.Label(left, text="Baseline captured. Set your session target.",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text2"]).pack(anchor="w", pady=(2,24))

        # Baseline stats
        stats = tk.Frame(left, bg=COLORS["surface"],
                          highlightthickness=1,
                          highlightbackground=COLORS["border"])
        stats.pack(fill="x", pady=(0,20))
        tk.Label(stats, text="BASELINE", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]
                 ).grid(row=0, column=0, sticky="w", padx=12, pady=8)
        self.cal_mean_lbl = tk.Label(stats, text="mean: —",
                                      font=FONT_LABEL, bg=COLORS["surface"],
                                      fg=COLORS["accent2"])
        self.cal_mean_lbl.grid(row=0, column=1, padx=12, pady=8)
        self.cal_sigma_lbl = tk.Label(stats, text="σ: —",
                                       font=FONT_LABEL, bg=COLORS["surface"],
                                       fg=COLORS["accent2"])
        self.cal_sigma_lbl.grid(row=0, column=2, padx=12, pady=8)

        # Rep target
        tk.Label(left, text="REP TARGET  (leave blank = run until fatigue)",
                 font=FONT_LABEL, bg=COLORS["bg"],
                 fg=COLORS["text3"]).pack(anchor="w", pady=(0,4))
        entry_row = tk.Frame(left, bg=COLORS["bg"])
        entry_row.pack(anchor="w", pady=(0,24))
        self.rep_target_entry = tk.Entry(
            entry_row, font=FONT_SMALL,
            bg=COLORS["surface2"], fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief="flat", width=8,
            highlightthickness=1,
            highlightbackground=COLORS["border"])
        self.rep_target_entry.pack(side="left", padx=(0,8), ipady=6)
        tk.Label(entry_row, text="reps", font=FONT_LABEL,
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(side="left")

        self.btn_start_session = self._make_button(
            left, "[ START SESSION → ]",
            command=self._on_start_session,
            color=COLORS["green"], width=26)
        self.btn_start_session.pack(anchor="w", pady=4)

        # Live preview
        right = tk.Frame(f, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True, padx=24, pady=32)
        tk.Label(right, text="LIVE SIGNAL PREVIEW",
                 font=FONT_LABEL, bg=COLORS["bg"],
                 fg=COLORS["text3"]).pack(anchor="w", pady=(0,6))
        self._build_mini_plot(right, "cal")

    # ── SCREEN: SESSION ───────────────────────

    def _build_session_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["SESSION"] = f

        top = tk.Frame(f, bg=COLORS["bg"])
        top.pack(fill="x", padx=24, pady=(20,8))

        # Rep counter
        rep_card = self._make_card(top, 200, 130)
        rep_card.pack(side="left", padx=(0,12))
        tk.Label(rep_card, text="REPS", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).place(x=16, y=12)
        self.rep_lbl = tk.Label(rep_card, text="0", font=FONT_BIG,
                                 bg=COLORS["surface"], fg=COLORS["green"])
        self.rep_lbl.place(relx=0.5, rely=0.55, anchor="center")

        # Z-score
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

        # Timer
        tim_card = self._make_card(top, 200, 130)
        tim_card.pack(side="left", padx=(0,12))
        tk.Label(tim_card, text="ELAPSED", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).place(x=16, y=12)
        self.timer_lbl = tk.Label(tim_card, text="00:00", font=FONT_MED,
                                   bg=COLORS["surface"], fg=COLORS["text"])
        self.timer_lbl.place(relx=0.5, rely=0.5, anchor="center")

        # Activation dot
        act_card = self._make_card(top, 160, 130)
        act_card.pack(side="left", padx=(0,12))
        tk.Label(act_card, text="ACTIVATION", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).place(x=12, y=12)
        self.act_dot = tk.Label(act_card, text="●",
                                 font=("Courier New", 36),
                                 bg=COLORS["surface"], fg=COLORS["text3"])
        self.act_dot.place(relx=0.5, rely=0.58, anchor="center")

        # End session
        self._make_button(top, "[ END SESSION ]",
                           command=self._on_end_session,
                           color=COLORS["red"], width=16).pack(side="right")

        # Fatigue bar
        fat_frame = tk.Frame(f, bg=COLORS["bg"])
        fat_frame.pack(fill="x", padx=24, pady=(0,8))
        tk.Label(fat_frame, text="FATIGUE INDEX",
                 font=FONT_LABEL, bg=COLORS["bg"],
                 fg=COLORS["text3"]).pack(side="left", padx=(0,12))
        fat_bg = tk.Frame(fat_frame, bg=COLORS["surface2"], height=14,
                           highlightthickness=1,
                           highlightbackground=COLORS["border"])
        fat_bg.pack(side="left", fill="x", expand=True)
        self.fat_fill = tk.Frame(fat_bg, bg=COLORS["green"], height=14)
        self.fat_fill.place(x=0, y=0, relheight=1, relwidth=0)
        self.fat_lbl = tk.Label(fat_frame, text="0%",
                                 font=FONT_LABEL, bg=COLORS["bg"],
                                 fg=COLORS["text2"], width=5)
        self.fat_lbl.pack(side="left", padx=(8,0))

        self.fatigue_alert = tk.Label(f, text="",
                                       font=("Courier New", 11, "bold"),
                                       bg=COLORS["bg"], fg=COLORS["red"])
        self.fatigue_alert.pack()

        # Live plot
        plot_frame = tk.Frame(f, bg=COLORS["bg"])
        plot_frame.pack(fill="both", expand=True, padx=24, pady=(0,16))
        self._build_main_plot(plot_frame)

    # ── SCREEN: METRICS ───────────────────────

    def _build_metrics_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["METRICS"] = f

        tk.Label(f, text="SESSION COMPLETE",
                 font=("Courier New", 13, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(pady=(24,2))
        tk.Label(f, text="Performance summary",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text2"]).pack(pady=(0,16))

        cols = tk.Frame(f, bg=COLORS["bg"])
        cols.pack(fill="both", expand=True, padx=24)

        left = tk.Frame(cols, bg=COLORS["bg"], width=320)
        left.pack(side="left", fill="y", padx=(0,16))
        left.pack_propagate(False)
        self.metrics_frame = tk.Frame(left, bg=COLORS["bg"])
        self.metrics_frame.pack(fill="both", expand=True)

        right = tk.Frame(cols, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True)
        self._build_metrics_plot(right)

        self._make_button(f, "[ NEW SESSION ]",
                           command=self._on_new_session,
                           color=COLORS["accent"], width=18).pack(pady=16)

    # ── PLOTS ─────────────────────────────────

    def _build_mini_plot(self, parent, tag):
        fig, ax = plt.subplots(figsize=(5, 2.6), facecolor=COLORS["plot_bg"])
        ax.set_facecolor(COLORS["plot_bg"])
        ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(COLORS["border"])
        ax.set_ylabel("EMG", color=COLORS["text3"], fontsize=7,
                       fontfamily="monospace")
        ax.set_xlabel("samples", color=COLORS["text3"], fontsize=7,
                       fontfamily="monospace")
        line, = ax.plot([], [], color=COLORS["accent"], linewidth=1.2)
        fig.tight_layout(pad=0.8)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, f"mini_ax_{tag}",     ax)
        setattr(self, f"mini_line_{tag}",   line)
        setattr(self, f"mini_canvas_{tag}", canvas)

    def _build_main_plot(self, parent):
        fig, ax = plt.subplots(figsize=(8, 2.8), facecolor=COLORS["plot_bg"])
        ax.set_facecolor(COLORS["plot_bg"])
        ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(COLORS["border"])
        ax.set_ylabel("EMG", color=COLORS["text3"], fontsize=7,
                       fontfamily="monospace")
        ax.set_xlabel("samples", color=COLORS["text3"], fontsize=7,
                       fontfamily="monospace")
        self.main_line, = ax.plot([], [], color=COLORS["accent2"],
                                   linewidth=1.2)
        fig.tight_layout(pad=0.8)
        self.main_canvas = FigureCanvasTkAgg(fig, master=parent)
        self.main_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.main_ax = ax

    def _build_metrics_plot(self, parent):
        self.met_fig, self.met_ax = plt.subplots(
            figsize=(5, 3.2), facecolor=COLORS["plot_bg"])
        self.met_ax.set_facecolor(COLORS["plot_bg"])
        self.met_ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for sp in self.met_ax.spines.values():
            sp.set_color(COLORS["border"])
        self.met_canvas = FigureCanvasTkAgg(self.met_fig, master=parent)
        self.met_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── HELPERS ───────────────────────────────

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
                      highlightthickness=1,
                      highlightbackground=COLORS["border"])
        f.pack_propagate(False)
        return f

    def _stat_row(self, parent, label, value, color=None):
        row = tk.Frame(parent, bg=COLORS["surface"],
                        highlightthickness=1,
                        highlightbackground=COLORS["border"])
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"],
                 width=22, anchor="w").pack(side="left", padx=12, pady=8)
        tk.Label(row, text=str(value), font=FONT_LABEL,
                 bg=COLORS["surface"],
                 fg=color or COLORS["text"]).pack(side="right", padx=12)

    # ── BUTTON ACTIONS ────────────────────────

    def _on_connect(self):
        self.connect_msg.config(text=f"Connecting to {SERIAL_PORT}…",
                                 fg=COLORS["yellow"])
        self.btn_connect.config(state="disabled")
        serial_reader.connect(
            on_success=lambda: self.after(0, self._on_connected),
            on_fail=lambda msg: self.after(0, lambda: self._on_connect_fail(msg)))

    def _on_connected(self):
        self.status_dot.config(fg=COLORS["green"])
        self.status_lbl.config(text="CONNECTED")
        self.connect_msg.config(text="Connected ✓", fg=COLORS["green"])
        self.after(600, lambda: self._show_screen("WARMUP"))

    def _on_connect_fail(self, msg):
        self.connect_msg.config(text=f"Failed: {msg}", fg=COLORS["red"])
        self.btn_connect.config(state="normal")

    def _on_start_session(self):
        target = self.rep_target_entry.get().strip()
        if target.isdigit() and int(target) > 0:
            serial_reader.send(target)
            time.sleep(0.1)
        serial_reader.send("s")
        with store._lock:
            store.session_active = True
            store.session_start  = time.time()
            store.rep_count      = 0
            store.rep_peaks      = []
            store.fatigue        = 0.0
            store.fatigued       = False
        self._show_screen("SESSION")

    def _on_end_session(self):
        serial_reader.send("q")
        with store._lock:
            store.session_active = False
        self._compute_metrics()
        self._show_screen("METRICS")

    def _on_new_session(self):
        with store._lock:
            store.rep_count      = 0
            store.rep_peaks      = []
            store.fatigue        = 0.0
            store.fatigued       = False
            store.session_start  = None
            store.session_active = False
            store.warmup_done    = False
            store.warmup_count   = 0
        self.fatigue_alert.config(text="")
        self._show_screen("WARMUP")

    # ── METRICS ───────────────────────────────

    def _compute_metrics(self):
        s     = store
        peaks = s.rep_peaks if s.rep_peaks else [0]
        dur   = round(time.time() - s.session_start, 1) \
                if s.session_start else 0
        mm, ss = divmod(int(dur), 60)

        s.metrics = {
            "Total reps":           s.rep_count,
            "Session duration":     f"{mm:02d}:{ss:02d}",
            "Avg peak z-score":     round(float(np.mean(peaks)), 2),
            "Peak consistency (σ)": round(float(np.std(peaks)), 2),
            "Fatigue detected":     "YES" if s.fatigued else "NO",
        }
        if len(peaks) >= 4:
            early = float(np.mean(peaks[:3]))
            late  = float(np.mean(peaks[-3:]))
            trend = round((late - early) / max(early, 0.01) * 100, 1)
            s.metrics["Effort trend"] = \
                f"{'▲' if trend > 0 else '▼'} {abs(trend)}%"

        for w in self.metrics_frame.winfo_children():
            w.destroy()
        color_map = {
            "Total reps":           COLORS["green"],
            "Session duration":     COLORS["text"],
            "Avg peak z-score":     COLORS["accent2"],
            "Peak consistency (σ)": COLORS["text"],
            "Fatigue detected":     COLORS["red"] if s.fatigued else COLORS["green"],
            "Effort trend":         COLORS["yellow"],
        }
        for k, v in s.metrics.items():
            self._stat_row(self.metrics_frame, k, v, color_map.get(k))

        self.met_ax.cla()
        self.met_ax.set_facecolor(COLORS["plot_bg"])
        if len(peaks) > 1:
            bars = self.met_ax.bar(range(1, len(peaks)+1), peaks,
                                    width=0.6, alpha=0.85)
            for i, bar in enumerate(bars):
                fade = i / max(len(bars)-1, 1)
                bar.set_facecolor(plt.cm.RdYlGn(1 - fade * 0.8))
        self.met_ax.set_xlabel("Rep", color=COLORS["text3"], fontsize=7,
                                fontfamily="monospace")
        self.met_ax.set_ylabel("Peak Z", color=COLORS["text3"], fontsize=7,
                                fontfamily="monospace")
        self.met_ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for sp in self.met_ax.spines.values():
            sp.set_color(COLORS["border"])
        self.met_fig.tight_layout(pad=0.8)
        self.met_canvas.draw_idle()

    # ── GUI UPDATE LOOP ───────────────────────

    def _update_loop(self):
        s   = store
        scr = s.screen

        with s._lock:
            hist         = list(s.emg_history)
            z            = s.z_score
            act          = s.act
            reps         = s.rep_count
            fatigue      = s.fatigue
            fatigued     = s.fatigued
            warmup_count = s.warmup_count
            warmup_done  = s.warmup_done
            bl_mean      = s.baseline_mean
            bl_sigma     = s.baseline_sigma
            sess_start   = s.session_start
            sess_active  = s.session_active

        # WARMUP
        if scr == "WARMUP":
            self.warmup_count_lbl.config(
                text=f"{warmup_count} / {WARMUP_TOTAL}")
            self.warmup_fill.place(
                relwidth=min(1.0, warmup_count / WARMUP_TOTAL))
            if warmup_done:
                if bl_mean  is not None:
                    self.cal_mean_lbl.config(text=f"mean: {round(bl_mean, 1)}")
                if bl_sigma is not None:
                    self.cal_sigma_lbl.config(text=f"σ: {round(bl_sigma, 2)}")
                self._show_screen("CALIBRATE")

        # CALIBRATE live preview
        if scr == "CALIBRATE" and len(hist) > 1:
            self.mini_line_cal.set_data(range(len(hist)), hist)
            self.mini_ax_cal.set_xlim(0, len(hist))
            self.mini_ax_cal.set_ylim(max(0, min(hist)-50), max(hist)+50)
            self.mini_canvas_cal.draw_idle()

        # SESSION
        if scr == "SESSION":
            self.rep_lbl.config(text=str(reps))

            self.z_lbl.config(
                text=f"{z:+.2f}",
                fg=COLORS["red"] if z > 2.5 else COLORS["accent2"])
            self.z_bar_fill.place(
                width=min(176, max(0, int(abs(z) / 6.0 * 176))))

            self.act_dot.config(
                fg=COLORS["green"] if act else COLORS["text3"])

            if sess_start:
                elapsed = int(time.time() - sess_start)
                mm, ss  = divmod(elapsed, 60)
                self.timer_lbl.config(text=f"{mm:02d}:{ss:02d}")

            color = COLORS["green"] if fatigue < 0.5 else \
                    COLORS["yellow"] if fatigue < 0.8 else COLORS["red"]
            self.fat_fill.place(relwidth=min(1.0, fatigue))
            self.fat_fill.config(bg=color)
            self.fat_lbl.config(text=f"{int(fatigue*100)}%")

            if fatigued:
                self.fatigue_alert.config(
                    text="⚠  FATIGUE DETECTED — consider resting")

            # Auto-transition when Arduino ends session
            if not sess_active and sess_start is not None:
                self._compute_metrics()
                self._show_screen("METRICS")

            if len(hist) > 1:
                self.main_line.set_data(range(len(hist)), hist)
                self.main_ax.set_xlim(0, len(hist))
                self.main_ax.set_ylim(max(0, min(hist)-50), max(hist)+80)
                self.main_canvas.draw_idle()

        self.after(80, self._update_loop)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = ReSync()
    app.mainloop()
        