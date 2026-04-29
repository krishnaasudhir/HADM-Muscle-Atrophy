# -*- coding: utf-8 -*-

"""
ReSync — EMG Thigh Activation Monitor
======================================
Visual prototype with simulated data.
No hardware needed to run.

Install dependencies:
    pip install matplotlib numpy

Run:
    python resync_app.py
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import numpy as np
import threading
import time
import math
import random

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

COLORS = {
    "bg":           "#0d0d14",
    "surface":      "#13131f",
    "surface2":     "#1a1a2e",
    "border":       "#2a2a45",
    "accent":       "#7c6af7",
    "accent2":      "#4fc3f7",
    "green":        "#4ade80",
    "yellow":       "#fbbf24",
    "red":          "#f87171",
    "text":         "#e8e8f0",
    "text2":        "#8888aa",
    "text3":        "#4a4a6a",
    "plot_bg":      "#0a0a12",
}

FONT_TITLE  = ("Courier New", 28, "bold")
FONT_MONO   = ("Courier New", 11)
FONT_LABEL  = ("Courier New", 9)
FONT_BIG    = ("Courier New", 48, "bold")
FONT_MED    = ("Courier New", 20, "bold")
FONT_SMALL  = ("Courier New", 10)
FONT_BTN    = ("Courier New", 11, "bold")

SCREENS = ["CONNECT", "CALIBRATE", "SESSION", "METRICS"]

WARMUP_SECS   = 5
CONTRACT_SECS = 5
HISTORY_LEN   = 300

# ─────────────────────────────────────────────
# SIMULATED DATA ENGINE
# ─────────────────────────────────────────────

class Simulator:
    def __init__(self, store):
        self.store   = store
        self.t       = 0
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            s = self.store
            t = self.t

            # Base resting signal
            noise    = random.gauss(0, 15)
            baseline = 512 + noise

            # During session — simulate reps every ~3 seconds
            if s.screen == "SESSION":
                rep_phase = (t % 150) / 150.0
                if 0.1 < rep_phase < 0.5:
                    activation = 600 * math.sin(math.pi * (rep_phase - 0.1) / 0.4)
                    val = baseline + activation
                    if rep_phase > 0.45 and not s._rep_flagged:
                        s.rep_count  += 1
                        s._rep_flagged = True
                        s.rep_peaks.append(val)
                    if rep_phase < 0.1:
                        s._rep_flagged = False
                else:
                    val = baseline
                    s._rep_flagged = False

                # Fatigue creeps up over time
                s.fatigue = min(1.0, t / 1800.0)

            elif s.screen == "CALIBRATE" and s.cal_phase == "contract":
                val = baseline + 550 + random.gauss(0, 20)
            else:
                val = baseline

            val = max(0, min(1023, val))
            s.emg_raw = int(val)
            s.emg_history.append(int(val))
            if len(s.emg_history) > HISTORY_LEN:
                s.emg_history.pop(0)

            # Z-score
            if len(s.emg_history) > 20:
                arr  = np.array(s.emg_history[-100:])
                mu   = np.mean(arr)
                sig  = max(np.std(arr), 0.5)
                s.z_score = (val - mu) / sig
                s.mu      = mu
                s.sigma   = sig

            self.t += 1
            time.sleep(0.02)


# ─────────────────────────────────────────────
# APP STATE
# ─────────────────────────────────────────────

class Store:
    def __init__(self):
        self.screen        = "CONNECT"
        self.connected     = False
        self.emg_raw       = 0
        self.emg_history   = []
        self.z_score       = 0.0
        self.mu            = 512.0
        self.sigma         = 1.0
        self.rep_count     = 0
        self.rep_peaks     = []
        self.fatigue       = 0.0
        self.cal_phase     = "idle"   # idle | rest | contract | done
        self.cal_progress  = 0.0
        self.threshold     = None
        self.session_start = None
        self._rep_flagged  = False
        self.metrics       = {}

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

        self.store = Store()
        self.sim   = Simulator(self.store)

        self._build_ui()
        self._update_loop()

    # ─────────────────────────────────────────
    # TOP-LEVEL LAYOUT
    # ─────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self, bg=COLORS["bg"], height=64)
        hdr.pack(fill="x", padx=0, pady=0)
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
                                    font=FONT_LABEL, bg=COLORS["bg"],
                                    fg=COLORS["text2"])
        self.status_lbl.pack(side="right", pady=16)

        # Divider
        tk.Frame(self, bg=COLORS["border"], height=1).pack(fill="x")

        # ── Content area ──
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

        # Centre column
        col = tk.Frame(f, bg=COLORS["bg"])
        col.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(col, text="DEVICE SETUP", font=("Courier New", 13, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(pady=(0, 4))
        tk.Label(col, text="Connect your ReSync cuff\nto begin monitoring",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text2"], justify="center").pack(pady=(0, 32))

        # Big connect button
        self.btn_connect = self._make_button(
            col, "[ CONNECT BLUETOOTH ]",
            command=self._on_connect,
            color=COLORS["accent"], width=28
        )
        self.btn_connect.pack(pady=8)

        self.connect_msg = tk.Label(col, text="",
                                     font=FONT_LABEL, bg=COLORS["bg"],
                                     fg=COLORS["text2"])
        self.connect_msg.pack(pady=4)

        # Device info box
        info = tk.Frame(col, bg=COLORS["surface"], bd=0,
                         highlightthickness=1,
                         highlightbackground=COLORS["border"])
        info.pack(pady=20, fill="x")

        rows = [
            ("DEVICE",    "HC-05 Bluetooth Module"),
            ("BAUD",      "9600"),
            ("SENSOR",    "MyoWare 2.0 EMG"),
            ("PLACEMENT", "Rectus Femoris (Thigh)"),
        ]
        for i, (k, v) in enumerate(rows):
            bg = COLORS["surface"] if i % 2 == 0 else COLORS["surface2"]
            row = tk.Frame(info, bg=bg)
            row.pack(fill="x")
            tk.Label(row, text=k, font=FONT_LABEL,
                     bg=bg, fg=COLORS["text3"], width=12,
                     anchor="w").pack(side="left", padx=12, pady=6)
            tk.Label(row, text=v, font=FONT_LABEL,
                     bg=bg, fg=COLORS["text"], anchor="w").pack(side="left")

    # ─────────────────────────────────────────
    # SCREEN: CALIBRATE
    # ─────────────────────────────────────────

    def _build_calibrate_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["CALIBRATE"] = f

        # Left — instructions
        left = tk.Frame(f, bg=COLORS["bg"], width=380)
        left.pack(side="left", fill="y", padx=(32, 0), pady=32)
        left.pack_propagate(False)

        tk.Label(left, text="CALIBRATION", font=("Courier New", 13, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(anchor="w")
        tk.Label(left, text="Personalise your threshold",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text2"]).pack(anchor="w", pady=(2, 24))

        # Phase indicator
        self.cal_phase_lbl = tk.Label(left, text="READY TO CALIBRATE",
                                       font=FONT_MED,
                                       bg=COLORS["bg"], fg=COLORS["text"],
                                       wraplength=340, justify="left")
        self.cal_phase_lbl.pack(anchor="w", pady=(0, 8))

        self.cal_instruction = tk.Label(
            left,
            text="Press Start Calibration below.\nYou will be guided through\na rest phase and a max contraction.",
            font=FONT_SMALL, bg=COLORS["bg"],
            fg=COLORS["text2"], justify="left")
        self.cal_instruction.pack(anchor="w", pady=(0, 24))

        # Progress bar
        tk.Label(left, text="PROGRESS", font=FONT_LABEL,
                 bg=COLORS["bg"], fg=COLORS["text3"]).pack(anchor="w")
        prog_bg = tk.Frame(left, bg=COLORS["surface2"], height=8,
                            highlightthickness=1,
                            highlightbackground=COLORS["border"])
        prog_bg.pack(fill="x", pady=(4, 20))
        self.cal_prog_fill = tk.Frame(prog_bg, bg=COLORS["accent"], height=8)
        self.cal_prog_fill.place(x=0, y=0, relheight=1, relwidth=0)

        # Countdown
        self.cal_countdown = tk.Label(left, text="",
                                       font=FONT_BIG,
                                       bg=COLORS["bg"], fg=COLORS["accent"])
        self.cal_countdown.pack(anchor="w", pady=(0, 24))

        self.btn_start_cal = self._make_button(
            left, "[ START CALIBRATION ]",
            command=self._on_start_cal,
            color=COLORS["accent"], width=26
        )
        self.btn_start_cal.pack(anchor="w", pady=4)

        self.btn_to_session = self._make_button(
            left, "[ START SESSION → ]",
            command=lambda: self._show_screen("SESSION"),
            color=COLORS["green"], width=26
        )
        # Hidden until cal done

        # Right — live signal
        right = tk.Frame(f, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True, padx=24, pady=32)
        self._build_mini_plot(right, "cal")

    # ─────────────────────────────────────────
    # SCREEN: SESSION
    # ─────────────────────────────────────────

    def _build_session_screen(self):
        f = tk.Frame(self.content, bg=COLORS["bg"])
        self._screens["SESSION"] = f

        # ── Top row: rep counter + z-score + fatigue ──
        top = tk.Frame(f, bg=COLORS["bg"])
        top.pack(fill="x", padx=24, pady=(20, 8))

        # Rep counter card
        rep_card = self._make_card(top, width=200, height=130)
        rep_card.pack(side="left", padx=(0, 12))
        tk.Label(rep_card, text="REPS", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).place(x=16, y=12)
        self.rep_lbl = tk.Label(rep_card, text="0", font=FONT_BIG,
                                 bg=COLORS["surface"], fg=COLORS["green"])
        self.rep_lbl.place(relx=0.5, rely=0.55, anchor="center")

        # Z-score card
        z_card = self._make_card(top, width=200, height=130)
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

        # Session timer card
        tim_card = self._make_card(top, width=200, height=130)
        tim_card.pack(side="left", padx=(0, 12))
        tk.Label(tim_card, text="ELAPSED", font=FONT_LABEL,
                 bg=COLORS["surface"], fg=COLORS["text3"]).place(x=16, y=12)
        self.timer_lbl = tk.Label(tim_card, text="00:00", font=FONT_MED,
                                   bg=COLORS["surface"], fg=COLORS["text"])
        self.timer_lbl.place(relx=0.5, rely=0.5, anchor="center")

        # End session button (right side)
        btn_frame = tk.Frame(top, bg=COLORS["bg"])
        btn_frame.pack(side="right")
        self._make_button(btn_frame, "[ END SESSION ]",
                           command=self._on_end_session,
                           color=COLORS["red"], width=18).pack()

        # ── Fatigue bar ──
        fat_frame = tk.Frame(f, bg=COLORS["bg"])
        fat_frame.pack(fill="x", padx=24, pady=(0, 8))

        tk.Label(fat_frame, text="FATIGUE INDEX",
                 font=FONT_LABEL, bg=COLORS["bg"],
                 fg=COLORS["text3"]).pack(side="left", padx=(0, 12))

        fat_bg = tk.Frame(fat_frame, bg=COLORS["surface2"], height=14,
                           highlightthickness=1,
                           highlightbackground=COLORS["border"])
        fat_bg.pack(side="left", fill="x", expand=True)
        self.fat_fill = tk.Frame(fat_bg, bg=COLORS["green"], height=14)
        self.fat_fill.place(x=0, y=0, relheight=1, relwidth=0)

        self.fat_lbl = tk.Label(fat_frame, text="0%",
                                 font=FONT_LABEL, bg=COLORS["bg"],
                                 fg=COLORS["text2"], width=5)
        self.fat_lbl.pack(side="left", padx=(8, 0))

        # ── Live plot ──
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
        tk.Label(f, text="Performance summary",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text2"]).pack(pady=(0, 16))

        # Two column layout
        cols = tk.Frame(f, bg=COLORS["bg"])
        cols.pack(fill="both", expand=True, padx=24)

        # Left — stats
        left = tk.Frame(cols, bg=COLORS["bg"], width=320)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.pack_propagate(False)

        self.metrics_frame = tk.Frame(left, bg=COLORS["bg"])
        self.metrics_frame.pack(fill="both", expand=True)

        # Right — chart
        right = tk.Frame(cols, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True)
        self._build_metrics_plot(right)

        # Bottom buttons
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
        ax.set_ylabel("EMG", color=COLORS["text3"], fontsize=7,
                       fontfamily="monospace")
        ax.set_xlabel("samples", color=COLORS["text3"], fontsize=7,
                       fontfamily="monospace")

        line, = ax.plot([], [], color=COLORS["accent"], linewidth=1.2,
                         alpha=0.9)
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
        ax.set_ylabel("EMG", color=COLORS["text3"], fontsize=7,
                       fontfamily="monospace")
        ax.set_xlabel("samples", color=COLORS["text3"], fontsize=7,
                       fontfamily="monospace")

        self.main_line, = ax.plot([], [], color=COLORS["accent2"],
                                   linewidth=1.2, alpha=0.9)
        self.thresh_line = ax.axhline(y=0, color=COLORS["yellow"],
                                       linewidth=1, linestyle="--",
                                       alpha=0.6, visible=False)
        fig.tight_layout(pad=0.8)

        self.main_canvas = FigureCanvasTkAgg(fig, master=parent)
        self.main_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.main_ax = ax

    def _build_metrics_plot(self, parent):
        self.met_fig, self.met_ax = plt.subplots(
            figsize=(5, 3.2), facecolor=COLORS["plot_bg"])
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
        btn = tk.Button(
            parent, text=text, command=command,
            font=FONT_BTN, fg=color, bg=COLORS["surface"],
            activeforeground=COLORS["bg"], activebackground=color,
            relief="flat", bd=0, cursor="hand2",
            highlightthickness=1, highlightbackground=color,
            width=width, pady=8
        )
        return btn

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

    # ─────────────────────────────────────────
    # BUTTON ACTIONS
    # ─────────────────────────────────────────

    def _on_connect(self):
        self.connect_msg.config(text="Connecting...", fg=COLORS["yellow"])
        self.btn_connect.config(state="disabled")
        self.sim.start()

        def finish():
            time.sleep(1.5)   # simulate connection delay
            self.store.connected = True
            self.after(0, lambda: [
                self.connect_msg.config(
                    text="Connected ✓  Redirecting...",
                    fg=COLORS["green"]),
                self.after(800, lambda: self._show_screen("CALIBRATE"))
            ])
        threading.Thread(target=finish, daemon=True).start()

    def _on_start_cal(self):
        self.btn_start_cal.config(state="disabled")
        self.store.cal_phase = "rest"
        self._run_cal_rest()

    def _run_cal_rest(self):
        self.cal_phase_lbl.config(text="PHASE 1 — REST",
                                   fg=COLORS["accent2"])
        self.cal_instruction.config(
            text="Relax your thigh muscle completely.\nDo not move.")
        self._cal_countdown(WARMUP_SECS, color=COLORS["accent2"],
                             on_done=self._run_cal_contract)

    def _run_cal_contract(self):
        self.store.cal_phase = "contract"
        self.cal_phase_lbl.config(text="PHASE 2 — MAX CONTRACTION",
                                   fg=COLORS["yellow"])
        self.cal_instruction.config(
            text="Contract your thigh as hard as you can\nand hold it!")
        self._cal_countdown(CONTRACT_SECS, color=COLORS["yellow"],
                             on_done=self._cal_done)

    def _cal_countdown(self, secs, color, on_done):
        start = time.time()
        total = secs

        def tick():
            elapsed = time.time() - start
            remaining = max(0, total - elapsed)
            pct = min(1.0, elapsed / total)

            self.cal_countdown.config(
                text=str(math.ceil(remaining)) if remaining > 0 else "0",
                fg=color)
            self.cal_prog_fill.place(relwidth=pct)

            if elapsed < total:
                self.after(50, tick)
            else:
                on_done()

        tick()

    def _cal_done(self):
        self.store.cal_phase = "done"
        # Compute threshold from collected history
        if len(self.store.emg_history) > 10:
            arr = np.array(self.store.emg_history[-50:])
            self.store.threshold = float(np.mean(arr) + 2.5 * np.std(arr))

        self.cal_phase_lbl.config(text="CALIBRATION COMPLETE",
                                   fg=COLORS["green"])
        self.cal_instruction.config(
            text="Threshold saved. Press Start Session\nwhen you are ready.")
        self.cal_countdown.config(text="✓", fg=COLORS["green"])
        self.btn_to_session.pack(anchor="w", pady=4)

    def _on_end_session(self):
        self.store.screen = "METRICS"
        self._compute_metrics()
        self._show_screen("METRICS")

    def _on_new_session(self):
        s = self.store
        s.rep_count   = 0
        s.rep_peaks   = []
        s.fatigue     = 0.0
        s.session_start = None
        s.cal_phase   = "idle"
        s.cal_progress = 0.0
        self.btn_start_cal.config(state="normal")
        self.btn_to_session.pack_forget()
        self.cal_phase_lbl.config(text="READY TO CALIBRATE",
                                   fg=COLORS["text"])
        self.cal_instruction.config(
            text="Press Start Calibration below.\nYou will be guided through\na rest phase and a max contraction.")
        self.cal_countdown.config(text="")
        self.cal_prog_fill.place(relwidth=0)
        self._show_screen("CALIBRATE")

    def _compute_metrics(self):
        s = self.store
        peaks = s.rep_peaks if s.rep_peaks else [0]
        duration = round(time.time() - s.session_start, 1) \
            if s.session_start else 0

        s.metrics = {
            "Total reps":              s.rep_count,
            "Session duration":        f"{duration}s",
            "Avg peak activation":     round(float(np.mean(peaks)), 1),
            "Peak consistency (σ)":    round(float(np.std(peaks)), 1),
            "Fatigue index":           f"{round(s.fatigue * 100)}%",
        }

        # Populate stat rows
        for w in self.metrics_frame.winfo_children():
            w.destroy()

        colors_map = {
            "Total reps":           COLORS["green"],
            "Session duration":     COLORS["text"],
            "Avg peak activation":  COLORS["accent2"],
            "Peak consistency (σ)": COLORS["text"],
            "Fatigue index":        COLORS["yellow"],
        }
        for k, v in s.metrics.items():
            self._stat_row(self.metrics_frame, k, v, colors_map.get(k))

        # Rep peaks bar chart
        self.met_ax.cla()
        self.met_ax.set_facecolor(COLORS["plot_bg"])
        if len(peaks) > 1:
            xs = range(1, len(peaks) + 1)
            bars = self.met_ax.bar(xs, peaks, color=COLORS["accent"],
                                    width=0.6, alpha=0.85)
            # Colour last few bars redder to show fatigue
            for i, bar in enumerate(bars):
                fade = i / max(len(bars) - 1, 1)
                bar.set_facecolor(plt.cm.RdYlGn(1 - fade * 0.8))

            if s.threshold:
                self.met_ax.axhline(s.threshold, color=COLORS["yellow"],
                                     linestyle="--", linewidth=1,
                                     label="threshold")
                self.met_ax.legend(facecolor=COLORS["surface"],
                                    labelcolor=COLORS["text2"], fontsize=7)

        self.met_ax.set_xlabel("Rep", color=COLORS["text3"], fontsize=7,
                                fontfamily="monospace")
        self.met_ax.set_ylabel("Peak EMG", color=COLORS["text3"], fontsize=7,
                                fontfamily="monospace")
        self.met_ax.tick_params(colors=COLORS["text3"], labelsize=7)
        for spine in self.met_ax.spines.values():
            spine.set_color(COLORS["border"])
        self.met_fig.tight_layout(pad=0.8)
        self.met_canvas.draw_idle()

    # ─────────────────────────────────────────
    # GUI UPDATE LOOP  (every 80ms)
    # ─────────────────────────────────────────

    def _update_loop(self):
        s     = self.store
        scr   = s.screen
        hist  = list(s.emg_history)

        # ── Status dot ──
        if s.connected:
            self.status_dot.config(fg=COLORS["green"])
            self.status_lbl.config(text="CONNECTED")

        # ── Calibration mini plot ──
        if scr == "CALIBRATE" and len(hist) > 1:
            ln = self.mini_line_cal
            ln.set_data(range(len(hist)), hist)
            ax = self.mini_ax_cal
            ax.set_xlim(0, len(hist))
            ax.set_ylim(max(0, min(hist) - 50), max(hist) + 50)
            self.mini_canvas_cal.draw_idle()

        # ── Session ──
        if scr == "SESSION":
            if s.session_start is None:
                s.session_start = time.time()

            # Rep label
            self.rep_lbl.config(text=str(s.rep_count))

            # Z-score
            z = s.z_score
            self.z_lbl.config(
                text=f"{z:+.2f}",
                fg=COLORS["red"] if z > 2.5 else COLORS["accent2"])
            zw = min(176, max(0, int(abs(z) / 6.0 * 176)))
            self.z_bar_fill.place(width=zw)

            # Timer
            elapsed = int(time.time() - s.session_start)
            mm, ss  = divmod(elapsed, 60)
            self.timer_lbl.config(text=f"{mm:02d}:{ss:02d}")

            # Fatigue bar
            pct = s.fatigue
            color = COLORS["green"] if pct < 0.5 else \
                    COLORS["yellow"] if pct < 0.8 else COLORS["red"]
            self.fat_fill.place(relwidth=pct)
            self.fat_fill.config(bg=color)
            self.fat_lbl.config(text=f"{int(pct*100)}%")

            # Main EMG plot
            if len(hist) > 1:
                self.main_line.set_data(range(len(hist)), hist)
                self.main_ax.set_xlim(0, len(hist))
                self.main_ax.set_ylim(
                    max(0, min(hist) - 50), max(hist) + 80)
                if s.threshold:
                    self.thresh_line.set_ydata([s.threshold, s.threshold])
                    self.thresh_line.set_visible(True)
                self.main_canvas.draw_idle()

        self.after(80, self._update_loop)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = ReSync()
    app.mainloop()