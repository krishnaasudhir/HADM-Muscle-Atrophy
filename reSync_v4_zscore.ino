// ─────────────────────────────────────────────
//  ReSync V4 — Z-Score Statistical Detection
//  Hardware: Arduino Uno + MyoWare 2.0 + single LED
//
//  PREDICTION METHOD: Online z-score with
//  Welford's algorithm for running mean + variance.
//
//  Instead of a fixed calibration peak, this version
//  continuously models the baseline signal distribution
//  (mean μ and standard deviation σ). Activation is
//  detected when the signal is Z_THRESH standard
//  deviations above the current baseline.
//
//  Key advantage: no calibration flex needed.
//  The baseline auto-builds during rest and FREEZES
//  during activation, so muscle fatigue, electrode
//  drift, and movement artifacts don't corrupt it.
//
//  INTENDED EXPERIENCE: user gets immediate feedback
//  from the moment they put the device on, and the
//  device remains reliable across a full 30-min session
//  without any manual recalibration.
// ─────────────────────────────────────────────

// ── Pins ──────────────────────────────────────
const int PIN_EMG     = A0;
const int PIN_LED     = 6;
const int PIN_RED_LED = 9;

// ── Z-score parameters ────────────────────────
const float Z_THRESH       = 2.5;   // activation threshold in std devs
const float Z_DEACT_THRESH = 1.2;   // hysteresis: deactivate below this z
const int   WARMUP_SAMPLES = 200;   // collect 200 rest samples before enabling (~4s)

// ── Welford online mean/variance ──────────────
// Uses integer-scaled math to avoid float overflow
// mean and M2 stored as floats (Uno handles it fine)
long    wCount = 0;
float   wMean  = 0.0;
float   wM2    = 0.0;

// ── Smoothing ─────────────────────────────────
const int SMOOTH_N = 8;
int       readings[SMOOTH_N];
int       readIdx  = 0;
long      total    = 0;

// ── Fatigue detection ─────────────────────────
const int   FATIGUE_BASELINE_REPS = 3;    // reps used to establish baseline effort
const float FATIGUE_RATIO         = 1.4;  // 40% above baseline peak z = fatigued
const int   MAX_REPS              = 20;

float repPeakZ[MAX_REPS];   // peak z-score recorded for each rep
float peakZThisRep = 0.0;   // running peak z during current activation

// ── Session control ───────────────────────────
enum State { WARMUP_STATE, WAIT_STATE, RUNNING_STATE, DONE_STATE };
State state      = WARMUP_STATE;
int   targetReps = 0;   // 0 = no limit, run until fatigue

// ── State ─────────────────────────────────────
bool  active     = false;
bool  prevActive = false;
bool  fatigued   = false;
int   repCount   = 0;

// ── LED pulse during warmup ───────────────────
unsigned long lastWarmupBlink = 0;


// ═════════════════════════════════════════════
//  SETUP
// ═════════════════════════════════════════════
void setup() {
  Serial.begin(9600);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  pinMode(PIN_RED_LED, OUTPUT);
  digitalWrite(PIN_RED_LED, LOW);

  for (int i = 0; i < SMOOTH_N; i++) readings[i] = 0;

  Serial.println("=== ReSync V4 — Z-Score Detection ===");
  Serial.println("Rest your muscle — building baseline...");
}


// ═════════════════════════════════════════════
//  WELFORD UPDATE
//  Called only when NOT active — baseline must
//  reflect resting signal, not activated signal.
// ═════════════════════════════════════════════
void welfordUpdate(float x) {
  wCount++;
  float delta  = x - wMean;
  wMean       += delta / (float)wCount;
  float delta2 = x - wMean;
  wM2         += delta * delta2;
}

float welfordStdDev() {
  if (wCount < 2) return 1.0;       // guard against divide-by-zero
  float variance = wM2 / (float)(wCount - 1);
  // Integer square root approximation — avoids <math.h> sqrt float issues
  // Use a simple Newton's method iteration
  if (variance <= 0) return 1.0;
  float s = variance;
  for (int i = 0; i < 8; i++) s = 0.5f * (s + variance / s);
  return s;
}

float computeZ(float x) {
  float sigma = welfordStdDev();
  if (sigma < 0.5) sigma = 0.5;     // floor — prevents extreme z on silent signal
  return (x - wMean) / sigma;
}


// ═════════════════════════════════════════════
//  MAIN LOOP
// ═════════════════════════════════════════════
void loop() {
  int raw      = analogRead(PIN_EMG);
  int smoothed = smooth(raw);

  // ── DONE: session over, nothing to do ───────
  if (state == DONE_STATE) return;

  // ── WARMUP: build resting baseline ──────────
  if (state == WARMUP_STATE) {
    welfordUpdate((float)smoothed);

    if (millis() - lastWarmupBlink > 400) {
      static bool warmLED = false;
      warmLED = !warmLED;
      setLED(warmLED ? 20 : 0);
      lastWarmupBlink = millis();
    }

    if (wCount >= WARMUP_SAMPLES) {
      setLED(0);
      blinkLED(255, 3);
      delay(500);
      state = WAIT_STATE;
      Serial.print("Baseline ready — mean=");
      Serial.print(wMean);
      Serial.print(" sigma=");
      Serial.println(welfordStdDev());
      Serial.println();
      Serial.println("Enter a rep target (e.g. '10'), then 's' to start.");
      Serial.println("Or just 's' to run until fatigue.");
      Serial.println("Type 'q' during a session to end early.");
    } else {
      Serial.print("Warmup "); Serial.print(wCount);
      Serial.print("/");       Serial.println(WARMUP_SAMPLES);
    }
    delay(20);
    return;
  }

  // ── WAIT: prompt for rep target, wait for 's' ─
  if (state == WAIT_STATE) {
    if (Serial.available()) {
      String input = Serial.readStringUntil('\n');
      input.trim();
      if (input.equalsIgnoreCase("s")) {
        state = RUNNING_STATE;
        Serial.print("Starting! ");
        if (targetReps > 0) {
          Serial.print("Target: "); Serial.print(targetReps); Serial.println(" reps.");
        } else {
          Serial.println("No rep limit — stops on fatigue.");
        }
      } else {
        int n = input.toInt();
        if (n > 0) {
          targetReps = n;
          Serial.print("Rep target set to "); Serial.print(targetReps);
          Serial.println(". Type 's' to start.");
        } else {
          Serial.println("Enter a number for reps, or 's' to start.");
        }
      }
    }
    return;
  }

  // ── RUNNING ──────────────────────────────────

  // Check for early quit
  if (Serial.available()) {
    char c = (char)Serial.read();
    if (c == 'q' || c == 'Q') {
      state = DONE_STATE;
      digitalWrite(PIN_RED_LED, HIGH);
      setLED(0);
      Serial.println("Session ended early.");
      return;
    }
  }

  float z = computeZ((float)smoothed);

  if (!active && z > Z_THRESH)       active = true;
  if ( active && z < Z_DEACT_THRESH) active = false;

  // Track peak z during each activation
  if (active && z > peakZThisRep) peakZThisRep = z;

  // Rising edge: count rep, check target-rep stop condition
  if (active && !prevActive) {
    if (repCount < MAX_REPS) repCount++;
    Serial.print("Rep: "); Serial.println(repCount);
    if (targetReps > 0 && repCount >= targetReps) {
      state = DONE_STATE;
      digitalWrite(PIN_RED_LED, HIGH);
      setLED(0);
      Serial.println("*** TARGET REPS REACHED — well done! ***");
      prevActive = active;
      return;
    }
  }

  // Falling edge: store peak z, check fatigue stop condition
  if (!active && prevActive) {
    int idx = repCount - 1;
    if (idx >= 0 && idx < MAX_REPS) repPeakZ[idx] = peakZThisRep;
    peakZThisRep = 0.0;

    if (!fatigued && repCount > FATIGUE_BASELINE_REPS) {
      float baselineAvg = 0.0;
      for (int i = 0; i < FATIGUE_BASELINE_REPS; i++) baselineAvg += repPeakZ[i];
      baselineAvg /= FATIGUE_BASELINE_REPS;

      Serial.print("peakZ="); Serial.print(repPeakZ[idx], 2);
      Serial.print(" baseline="); Serial.println(baselineAvg, 2);

      if (repPeakZ[idx] > baselineAvg * FATIGUE_RATIO) {
        fatigued = true;
        state = DONE_STATE;
        digitalWrite(PIN_RED_LED, HIGH);
        setLED(0);
        Serial.println("*** FATIGUE DETECTED — rest now ***");
        prevActive = active;
        return;
      }
    }
  }

  prevActive = active;

  // Only update baseline during rest
  if (!active) welfordUpdate((float)smoothed);

  // ── LED actuation ───────────────────────────
  if (active) {
    float zClamped = z < Z_THRESH ? Z_THRESH : (z > 7.0 ? 7.0 : z);
    int brightness = (int)map((long)(zClamped * 100),
                               (long)(Z_THRESH * 100), 700,
                               80, 255);
    setLED(constrain(brightness, 80, 255));
  } else {
    setLED(0);
  }

  Serial.print("raw=");  Serial.print(raw);
  Serial.print(" sm=");  Serial.print(smoothed);
  Serial.print(" μ=");   Serial.print(wMean, 1);
  Serial.print(" σ=");   Serial.print(welfordStdDev(), 1);
  Serial.print(" z=");   Serial.print(z, 2);
  Serial.print(" act="); Serial.println(active ? "YES" : "no");

  delay(20);
}


// ── Helpers ───────────────────────────────────
int smooth(int v) {
  total -= readings[readIdx];
  readings[readIdx] = v;
  total += v;
  readIdx = (readIdx + 1) % SMOOTH_N;
  return (int)(total / SMOOTH_N);
}

void setLED(int b)               { analogWrite(PIN_LED, b); }
void blinkLED(int b, int times)  {
  for (int i = 0; i < times; i++) { setLED(b); delay(250); setLED(0); delay(200); }
}
