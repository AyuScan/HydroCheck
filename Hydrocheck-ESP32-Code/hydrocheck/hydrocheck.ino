/*
  HYDROCHECK - Full Prototype
  Predictive Dehydration & Heat Stress Wearable

  Hardware:
    ESP32 DevKit
    TCA9548A I2C Multiplexer (0x70)
      -> Ch0: MAX30102 (PPG, HR)
      -> Ch1: MLX90614 (GY-906, skin/ambient IR temp)
      -> Ch2: SH1106 1.3" OLED (128x64)
    Grove GSR sensor -> GPIO34 (direct analog, NOT on mux)
    Buzzer/vibration motor -> GPIO25

  NOTE: Calibration now runs automatically ~1.5s after power-on.
  No button needed - just power the device and hold still for 15s.

  Libraries:
    SparkFun MAX3010x Pulse and Proximity Sensor Library
    Adafruit MLX90614
    Adafruit SH110X + Adafruit GFX
    BLEDevice (built into ESP32 core)
*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Adafruit_MLX90614.h>
#include "MAX30105.h"
#include "heartRate.h"

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ---------------- I2C mux ----------------
#define TCAADDR 0x70
void tcaSelect(uint8_t ch) {
  if (ch > 7) return;
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
}

// ---------------- Pins ----------------
#define GSR_PIN        34
#define BUZZER_PIN     25
const int GSR_SAMPLES = 10;

// ---------------- Devices ----------------
MAX30105 maxSensor;
Adafruit_MLX90614 mlx = Adafruit_MLX90614();
Adafruit_SH1106G display = Adafruit_SH1106G(128, 64, &Wire, -1);

// ---------------- BLE ----------------
#define SERVICE_UUID        "0000181A-0000-1000-8000-00805F9B34FB"
#define CHARACTERISTIC_UUID "00002A6E-0000-1000-8000-00805F9B34FB"
BLECharacteristic *pCharacteristic;
bool bleConnected = false;

class ServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* s) { bleConnected = true; }
  void onDisconnect(BLEServer* s) { bleConnected = false; BLEDevice::startAdvertising(); }
};

// ---------------- Timing / adaptive sampling ----------------
unsigned long lastSampleTime = 0;
unsigned long samplingIntervalMs = 2000;
const unsigned long MIN_INTERVAL_MS = 500;
const unsigned long MAX_INTERVAL_MS = 5000;

// ---------------- GSR Kalman + rolling window ----------------
const int GSR_WINDOW = 10;
float gsrBuffer[GSR_WINDOW];
int gsrIndex = 0;
bool gsrBufferFull = false;
float gsrKalmanEstimate = 0;
float gsrKalmanError = 1;
const float GSR_PROCESS_NOISE = 0.02;
const float GSR_MEASURE_NOISE = 4.0;

// ---------------- Baselines (set at calibration) ----------------
float baselineGSR = 0;
float baselineHR = 70;
float baselineSkinTemp = 0;
float restingIBIStd = 0;
bool calibrated = false;

// ---------------- HRV proxy ----------------
unsigned long lastBeatTime = 0;
float ibiBuffer[8] = {0};
int ibiIndex = 0;

// ---------------- Cross-cycle BPM smoothing ----------------
const int BPM_HISTORY_SIZE = 4;
float bpmHistory[BPM_HISTORY_SIZE] = {0};
int bpmHistoryIndex = 0;
int bpmHistoryCount = 0;

float smoothBpm(float newBpm) {
  if (newBpm <= 0) return -1; // signal "no update this cycle", caller keeps last good value
  bpmHistory[bpmHistoryIndex] = newBpm;
  bpmHistoryIndex = (bpmHistoryIndex + 1) % BPM_HISTORY_SIZE;
  if (bpmHistoryCount < BPM_HISTORY_SIZE) bpmHistoryCount++;

  float sum = 0;
  for (int i = 0; i < bpmHistoryCount; i++) sum += bpmHistory[i];
  return sum / bpmHistoryCount;
}

float lastGoodBpm = 0; // persists across cycles when a cycle catches 0 beats
int cyclesSinceGoodContact = 0;
const int MAX_HOLD_CYCLES = 2; // invalidate BPM if no good contact for this many cycles

// ---------------- Heat strain accumulator ----------------
float heatStrainScore = 0;
const float STRAIN_GAIN = 0.6;
const float STRAIN_RECOVERY = 0.4;

// ==================================================================
int readGSR() {
  long sum = 0;
  for (int i = 0; i < GSR_SAMPLES; i++) {
    sum += analogRead(GSR_PIN);
    delay(2);
  }
  return sum / GSR_SAMPLES;
}

float kalmanFilter(float measurement) {
  gsrKalmanError += GSR_PROCESS_NOISE;
  float kalmanGain = gsrKalmanError / (gsrKalmanError + GSR_MEASURE_NOISE);
  gsrKalmanEstimate += kalmanGain * (measurement - gsrKalmanEstimate);
  gsrKalmanError *= (1 - kalmanGain);
  return gsrKalmanEstimate;
}

float updateGSRWindow(float filteredValue) {
  float oldValue = gsrBuffer[gsrIndex];
  gsrBuffer[gsrIndex] = filteredValue;
  gsrIndex = (gsrIndex + 1) % GSR_WINDOW;
  if (gsrIndex == 0) gsrBufferFull = true;
  if (!gsrBufferFull) return 0;
  return (filteredValue - oldValue) / GSR_WINDOW;
}

float computeIBIStdDev() {
  float mean = 0;
  for (int i = 0; i < 8; i++) mean += ibiBuffer[i];
  mean /= 8;
  float variance = 0;
  for (int i = 0; i < 8; i++) variance += pow(ibiBuffer[i] - mean, 2);
  variance /= 8;
  return sqrt(variance);
}

// ==================================================================
void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);

  pinMode(BUZZER_PIN, OUTPUT);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  pinMode(GSR_PIN, INPUT);

  // MAX30102
  tcaSelect(0);
  if (maxSensor.begin(Wire)) {
    Serial.println("MAX30102 OK");
    maxSensor.setup();
    maxSensor.setPulseAmplitudeRed(0x3F);   // much stronger LED for wrist tissue
    maxSensor.setPulseAmplitudeIR(0x3F);
    maxSensor.setPulseAmplitudeGreen(0);
  } else {
    Serial.println("MAX30102 NOT FOUND");
  }

  // MLX90614
  tcaSelect(1);
  if (mlx.begin()) {
    Serial.println("MLX90614 OK");
  } else {
    Serial.println("MLX90614 NOT FOUND");
  }

  // OLED
  tcaSelect(2);
  if (display.begin(0x3C, true)) {
    Serial.println("OLED OK");
    drawBootScreen();
  } else {
    Serial.println("OLED NOT FOUND");
  }

  // BLE
  BLEDevice::init("Hydrocheck");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());
  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
      CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();
  pServer->getAdvertising()->start();
  Serial.println("BLE advertising started");

  delay(1500); // let the splash screen show briefly before calibration starts
  runCalibration(); // auto-calibrate immediately on power-up, no button needed
}

// ==================================================================
void drawBootScreen() {
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);

  // Title bar
  display.setTextSize(2);
  display.setCursor(6, 6);
  display.println("HYDRO");
  display.setCursor(6, 26);
  display.println("CHECK");

  // droplet icon (simple)
  int dx = 96, dy = 20;
  display.fillTriangle(dx, dy - 12, dx - 8, dy + 4, dx + 8, dy + 4, SH110X_WHITE);
  display.fillCircle(dx, dy + 6, 9, SH110X_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 50);
  display.println("Starting up...");
  display.display();
}

// ==================================================================
void runCalibration() {
  Serial.println("Calibrating... stay still for 15 seconds.");

  float gsrSum = 0, tempSum = 0;
  int hrSamples = 0;
  float hrSum = 0;
  unsigned long start = millis();
  const unsigned long CAL_DURATION_MS = 15000;
  int count = 0;
  float ibiCalBuffer[8] = {0};
  int ibiCalIndex = 0;

  while (millis() - start < CAL_DURATION_MS) {
    gsrSum += readGSR();

    tcaSelect(1);
    tempSum += mlx.readObjectTempC();
    count++;

    tcaSelect(0);
    long irValue = maxSensor.getIR();
    if (checkForBeat(irValue)) {
      unsigned long now = millis();
      long delta = now - lastBeatTime;
      lastBeatTime = now;
      if (delta > 250 && delta < 2000) {
        float bpm = 60000.0 / delta;
        hrSum += bpm;
        hrSamples++;
        ibiCalBuffer[ibiCalIndex % 8] = delta;
        ibiCalIndex++;
      }
    }

    // ---- animated calibration screen ----
    tcaSelect(2);
    float progress = (float)(millis() - start) / CAL_DURATION_MS;
    drawCalibrationScreen(progress);

    delay(150);
  }

  baselineGSR = gsrSum / count;
  baselineSkinTemp = tempSum / count;
  baselineHR = hrSamples > 0 ? (hrSum / hrSamples) : 70;

  float mean = 0;
  for (int i = 0; i < 8; i++) mean += ibiCalBuffer[i];
  mean /= 8;
  float variance = 0;
  for (int i = 0; i < 8; i++) variance += pow(ibiCalBuffer[i] - mean, 2);
  restingIBIStd = sqrt(variance / 8);

  calibrated = true;

  Serial.printf("Baseline GSR:%.2f HR:%.1f SkinTemp:%.2f IBIstd:%.2f\n",
                baselineGSR, baselineHR, baselineSkinTemp, restingIBIStd);

  tcaSelect(2);
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);
  display.setTextSize(2);
  display.setCursor(4, 20);
  display.println("READY");
  display.setTextSize(1);
  display.setCursor(6, 44);
  display.println("Baseline locked in");
  display.display();
  delay(1200);
}

// ---------------- Calibration screen with progress ring ----------------
void drawCalibrationScreen(float progress) {
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);

  display.setTextSize(1);
  display.setCursor(20, 2);
  display.println("CALIBRATING");
  display.drawFastHLine(0, 12, 128, SH110X_WHITE);

  // circular progress ring, center (64,38), radius 20
  int cx = 64, cy = 38, r = 20;
  int totalDeg = (int)(progress * 360);
  for (int a = 0; a < totalDeg; a += 4) {
    float rad = (a - 90) * PI / 180.0;
    int x = cx + r * cos(rad);
    int y = cy + r * sin(rad);
    display.fillCircle(x, y, 2, SH110X_WHITE);
  }

  display.setTextSize(1);
  char pctStr[6];
  snprintf(pctStr, sizeof(pctStr), "%d%%", (int)(progress * 100));
  int16_t x1, y1; uint16_t w, h;
  display.getTextBounds(pctStr, 0, 0, &x1, &y1, &w, &h);
  display.setCursor(cx - w / 2, cy - h / 2);
  display.println(pctStr);

  display.setCursor(4, 56);
  display.println("Hold still...");
  display.display();
}

// ==================================================================
void loop() {
  unsigned long now = millis();
  if (now - lastSampleTime >= samplingIntervalMs) {
    lastSampleTime = now;
    sampleAndProcess();
  }
}

// ==================================================================
// ---------------- Icon helpers (small, hand-drawn) ----------------
void drawHeartIcon(int x, int y) {
  display.fillCircle(x + 2, y + 2, 3, SH110X_WHITE);
  display.fillCircle(x + 7, y + 2, 3, SH110X_WHITE);
  display.fillTriangle(x - 1, y + 3, x + 10, y + 3, x + 4, y + 10, SH110X_WHITE);
}

void drawThermoIcon(int x, int y) {
  display.drawRoundRect(x, y, 5, 12, 2, SH110X_WHITE);
  display.fillCircle(x + 2, y + 14, 3, SH110X_WHITE);
  display.fillRect(x + 1, y + 6, 3, 8, SH110X_WHITE);
}

void drawBoltIcon(int x, int y) {
  display.drawLine(x + 4, y, x, y + 6, SH110X_WHITE);
  display.drawLine(x, y + 6, x + 3, y + 6, SH110X_WHITE);
  display.drawLine(x + 3, y + 6, x - 1, y + 13, SH110X_WHITE);
  display.drawLine(x + 4, y, x + 3, y + 6, SH110X_WHITE);
  display.drawLine(x + 3, y + 6, x + 6, y + 6, SH110X_WHITE);
  display.drawLine(x + 6, y + 6, x + 4, y, SH110X_WHITE);
}

// ---------------- Main live dashboard ----------------
void drawDashboard(float riskIndex, float bpm, float skinTemp, float strain, bool bleOn) {
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);

  // ---- header ----
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("HYDROCHECK");
  if (bleOn) {
    display.setCursor(112, 0);
    display.print((char)0x2A); // small marker to indicate BLE link active
  }
  display.drawFastHLine(0, 9, 128, SH110X_WHITE);

  // ---- risk level, large, left side ----
  String riskLabel = "LOW";
  if (riskIndex > 60) riskLabel = "HIGH";
  else if (riskIndex > 30) riskLabel = "MED";

  display.setTextSize(2);
  display.setCursor(0, 14);
  display.printf("%.0f%%", riskIndex);

  display.setTextSize(1);
  display.setCursor(0, 33);
  display.print(riskLabel);
  display.print(" RISK");

  // ---- risk bar (horizontal, under the label) ----
  int barX = 0, barY = 43, barW = 60, barH = 6;
  display.drawRect(barX, barY, barW, barH, SH110X_WHITE);
  int fillW = constrain((int)(riskIndex / 100.0 * (barW - 2)), 0, barW - 2);
  if (fillW > 0) display.fillRect(barX + 1, barY + 1, fillW, barH - 2, SH110X_WHITE);

  // ---- vertical divider ----
  display.drawFastVLine(68, 12, 40, SH110X_WHITE);

  // ---- right column: HR, Skin temp, Strain ----
  drawHeartIcon(74, 13);
  display.setTextSize(1);
  display.setCursor(88, 15);
  if (bpm > 0) {
    display.printf("%.0f", bpm);
  } else {
    display.print("--");
  }
  display.setCursor(88, 24);
  display.print(bpm > 0 ? "bpm" : "no skin");

  drawThermoIcon(76, 31);
  display.setCursor(88, 34);
  display.printf("%.1fC", skinTemp);

  drawBoltIcon(76, 48);
  display.setCursor(88, 50);
  display.printf("%.0f%%", strain);

  // ---- footer status bar ----
  display.drawFastHLine(0, 54, 128, SH110X_WHITE);
  display.setCursor(0, 57);
  display.print(riskIndex > 60 ? "ALERT: Rest advised" : "Status: Monitoring");

  display.display();
}

// ==================================================================
void sampleAndProcess() {
  // ---- GSR ----
  int rawGSR = readGSR();
  float filteredGSR = kalmanFilter(rawGSR);
  float gsrRateOfChange = updateGSRWindow(filteredGSR);

  // ---- MLX90614 ----
  tcaSelect(1);
  float skinTemp = mlx.readObjectTempC();
  float ambientTemp = mlx.readAmbientTempC();

  // ---- MAX30102: fast burst sampling for reliable beat detection ----
  tcaSelect(0);
  float bpm = 0;
  long lastIR = 0;
  unsigned long burstStart = millis();
  const unsigned long BURST_DURATION_MS = 5000; // longer burst = more beats to average
  float bpmReadings[6];
  int bpmCount = 0;
  int goodContactSamples = 0;
  int totalSamples = 0;

  while (millis() - burstStart < BURST_DURATION_MS) {
    long irValue = maxSensor.getIR();
    lastIR = irValue;
    totalSamples++;
    if (irValue > 50000) {   // require strong, confirmed contact
      goodContactSamples++;
      if (checkForBeat(irValue)) {
        unsigned long beatNow = millis();
        long delta = beatNow - lastBeatTime;
        lastBeatTime = beatNow;
        if (delta > 400 && delta < 1500) {   // plausible range: ~40-150 bpm only
          float instantBpm = 60000.0 / delta;
          if (bpmCount < 6) {
            bpmReadings[bpmCount] = instantBpm;
            bpmCount++;
          }
          ibiBuffer[ibiIndex % 8] = delta;
          ibiIndex++;
        }
      }
    }
    delay(10); // ~100Hz sampling, required for checkForBeat to trace the waveform
  }

  // require contact for at least 70% of the burst to trust this cycle at all
  bool goodContact = (goodContactSamples >= (int)(totalSamples * 0.7));

  // average the accepted beats from this burst, reject if too few for confidence
  float rawCycleBpm = 0;
  if (goodContact && bpmCount >= 2) {
    float sum = 0;
    for (int i = 0; i < bpmCount; i++) sum += bpmReadings[i];
    rawCycleBpm = sum / bpmCount;
  }

  if (!goodContact) {
    cyclesSinceGoodContact++;
  } else {
    cyclesSinceGoodContact = 0;
  }

  // cross-cycle smoothing: blend this cycle's reading with recent history
  float smoothed = smoothBpm(rawCycleBpm);
  if (smoothed > 0) {
    bpm = smoothed;
    lastGoodBpm = smoothed;
  } else if (cyclesSinceGoodContact < MAX_HOLD_CYCLES) {
    bpm = lastGoodBpm; // brief dropout - hold last known good value
  } else {
    // contact lost for too long - value is no longer trustworthy, reset everything
    bpm = 0;
    lastGoodBpm = 0;
    bpmHistoryCount = 0;
    bpmHistoryIndex = 0;
    ibiIndex = 0; // force HRV to also require a fresh full buffer of real beats
  }

  Serial.print("Raw IR: ");
  Serial.println(lastIR);   // >50000 = good contact, <5000 = no/poor contact
  float hrv = (ibiIndex >= 8) ? computeIBIStdDev() : 0;

  // ---- Risk scoring ----
  float gsrRisk = 0, hrvRisk = 0, tempRisk = 0;
  if (calibrated) {
    gsrRisk  = constrain(gsrRateOfChange * 20.0, 0, 40);
    hrvRisk  = (hrv > 0 && hrv < restingIBIStd * 0.6) ? 25 : 0;
    tempRisk = constrain((skinTemp - baselineSkinTemp) * 8.0, 0, 35);

    bool activeOrHot = (bpm > baselineHR + 15) || (tempRisk > 10);
    if (activeOrHot) {
      heatStrainScore += STRAIN_GAIN * (1 + tempRisk / 10.0);
    } else {
      heatStrainScore -= STRAIN_RECOVERY;
    }
    heatStrainScore = constrain(heatStrainScore, 0, 100);
  }

  float riskIndex = constrain(gsrRisk + hrvRisk + tempRisk * 0.5 + heatStrainScore * 0.3, 0, 100);

  // ---- Adaptive sampling ----
  samplingIntervalMs = map((int)riskIndex, 0, 100, MAX_INTERVAL_MS, MIN_INTERVAL_MS);
  samplingIntervalMs = constrain(samplingIntervalMs, MIN_INTERVAL_MS, MAX_INTERVAL_MS);

  // ---- Alert ----
  digitalWrite(BUZZER_PIN, riskIndex > 60 ? HIGH : LOW);

  // ---- Serial debug ----
  Serial.printf("GSR:%.1f dGSR:%.3f BPM:%.0f HRV:%.1f SkinT:%.1f Strain:%.1f Risk:%.1f Calib:%d\n",
                filteredGSR, gsrRateOfChange, bpm, hrv, skinTemp, heatStrainScore, riskIndex, calibrated);

  // ---- OLED dashboard ----
  tcaSelect(2);
  drawDashboard(riskIndex, bpm, skinTemp, heatStrainScore, bleConnected);

  // ---- BLE notify ----
  if (bleConnected) {
    char payload[128];
    snprintf(payload, sizeof(payload),
             "{\"risk\":%.1f,\"hr\":%.0f,\"skin\":%.1f,\"gsr\":%.0f,\"strain\":%.1f}",
             riskIndex, bpm, skinTemp, filteredGSR, heatStrainScore);
    pCharacteristic->setValue((uint8_t*)payload, strlen(payload));
    pCharacteristic->notify();
  }
}
