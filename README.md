# ThreshHold — EMG Workout Monitor

Real-time muscle activation tracker using an EMG sensor, Arduino, and Bluetooth.
Counts reps, detects fatigue, and streams live data to a desktop GUI — no manual calibration needed.

---

## Hardware Required

| Component | Notes |
|---|---|
| Arduino Uno | Any 5 V Uno clone works |
| MyoWare 2.0 EMG sensor | Use the ENV (envelope) output |
| HC-05 Bluetooth module | Classic BT, not BLE |
| Green LED + 220 Ω resistor | Activation indicator |
| Red LED + 220 Ω resistor | Fatigue / session-end indicator |
| 1 kΩ + 2 kΩ resistors | Voltage divider for HC-05 RX line |

---

## Wiring

### MyoWare 2.0 → Arduino
| MyoWare pin | Arduino pin |
|---|---|
| ENV | A0 |
| VCC | 5 V |
| GND | GND |

Place the two signal electrodes along the quad, reference electrode on a bony area (e.g. knee).

### HC-05 → Arduino
| HC-05 pin | Arduino pin |
|---|---|
| TX | 0 (RX) |
| RX | 1 (TX) via voltage divider* |
| VCC | 5 V |
| GND | GND |

*Voltage divider on RX: Arduino TX → 1 kΩ → HC-05 RX → 2 kΩ → GND.
This steps 5 V down to 3.3 V to protect the HC-05.

### LEDs
| LED | Arduino pin |
|---|---|
| Green LED (anode) | 6 |
| Red LED (anode) | 9 |
| Both cathodes | GND (through 220 Ω) |

---

## Software Setup

### 1. Arduino sketch
1. Open `reSync_v4_zscore.ino` in Arduino IDE.
2. **Disconnect HC-05 from pins 0/1** before uploading (the HC-05 blocks upload).
3. Upload to Arduino Uno.
4. Reconnect HC-05 to pins 0/1.

### 2. Python dependencies
```
pip install pyserial matplotlib numpy
```
`tkinter` is included with standard Python.

### 3. Pair HC-05 via Bluetooth
1. Power the Arduino (HC-05 LED blinks rapidly at first).
2. On Windows: Settings → Bluetooth → Add device → find "HC-05".
3. Pairing code: `1234` (or `0000`).
4. After pairing, open **Device Manager → Ports (COM & LPT)** — note the COM port labelled "Standard Serial over Bluetooth" (e.g. COM7).

---

## How to Use

### Every session (step by step)

1. **Power-cycle the Arduino** — unplug and replug USB or battery.
2. **Attach the EMG electrodes to your skin** before opening the GUI.
3. Open a terminal and run:
   ```
   python EMG_GUI.py
   ```
4. Enter your Bluetooth COM port (e.g. `COM7`) and click **CONNECT BLUETOOTH**.
5. Once connected, click **START CALIBRATION**.
   - Keep the target muscle completely relaxed.
   - The progress bar fills over ~4 seconds.
   - Check the terminal: `sigma` should be between 5 and 50.
     If you see `WARNING: sigma too large`, the sensor wasn't on skin — power-cycle and retry.
6. Optionally enter a rep target (leave blank to run until fatigue is detected).
7. Click **START SESSION →** and begin exercising.
8. The GUI shows live rep count, z-score, elapsed time, and fatigue index.
9. The session ends automatically when:
   - Target reps are reached, or
   - Fatigue is detected (peak effort rises 40 % above your baseline).
   You can also click **END SESSION** at any time.
10. Review the metrics screen, then click **NEW SESSION** to go again (no re-warmup needed).

---

## How It Works

**Z-score detection:** Instead of a fixed threshold, ThreshHold models your resting EMG signal as a statistical distribution (mean μ, standard deviation σ) using Welford's online algorithm. A contraction is detected when the signal exceeds μ + 2.5σ and deactivates below μ + 1.2σ (hysteresis).

**Baseline freeze:** The distribution only updates during rest — muscle activation pauses the Welford update, so fatigue and electrode drift don't corrupt the baseline.

**Fatigue detection:** The peak z-score of each rep is recorded. When a rep's peak exceeds 1.4× the average of the first 3 reps, fatigue is flagged and the session ends.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| GUI shows "DEVICE ALREADY RUNNING" | Power-cycle Arduino, then reconnect; in our case: unplug and replug the battery |
| `sigma > 150` after warmup | Electrodes weren't on skin during calibration — power-cycle and retry |
| No reps detected despite flexing | Re-do warmup; check sigma is < 50 in terminal |
| "Access Denied" on COM port | Close Arduino IDE Serial Monitor — it holds the port, or kill the terminal running EMG_GUI.py |
| No COM port appears after pairing | Open Control Panel → Devices → HC-05 → "More Bluetooth options" → COM Ports tab → add Outgoing port |
| Upload fails | Disconnect HC-05 from pins 0/1 before uploading sketch |
