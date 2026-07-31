# Hydrocheck Companion App

This directory contains the Capacitor-based React mobile application for the Hydrocheck predictive dehydration and heat stress monitoring system. The app interfaces with the Hydrocheck wearable device over Bluetooth Low Energy (BLE) to display live telemetry.

---

## 🛠️ Tech Stack & Features

*   **Frontend Core:** React with [Vite](https://vite.dev/) for quick development hot module reloading.
*   **Mobile Framework:** [Capacitor](https://capacitorjs.com/) to build cross-platform mobile packages (Android/iOS) using web assets.
*   **Bluetooth LE Integration:** `@capacitor-community/bluetooth-le` handles finding and connecting to the ESP32 GATT server, and subscribing to characteristics.
*   **Visualizations:** Responsive HTML5 Canvas component (`BpmChart`) showing real-time heart rate history.
*   **Activity Console:** Integrated debugging log showing real-time packets, status changes, and parsing info.

---

## 📁 File Structure

*   [App.jsx](file:///d:/Hydrocheck/Hydrocheck-app/src/App.jsx): Main application controller containing Bluetooth state handlers, telemetry parser, logs visualizer, and application layout.
*   [App.css](file:///d:/Hydrocheck/Hydrocheck-app/src/App.css): Dark theme layout stylesheet containing glassmorphic styling, progress rings, and animations.
*   [main.jsx](file:///d:/Hydrocheck/Hydrocheck-app/src/main.jsx): React entry point.
*   [capacitor.config.ts](file:///d:/Hydrocheck/Hydrocheck-app/capacitor.config.ts): Configuration file defining App ID, app name, and web assets directory.
*   [android/](file:///d:/Hydrocheck/Hydrocheck-app/android): Generated native Android Studio project.

---

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have [Node.js](https://nodejs.org/) and Android Studio (with Android SDK) installed.

### 2. Development Setup
```bash
npm install
```

### 3. Running in Dev Mode (Browser)
```bash
npm run dev
```
*Note: BLE is mocked or limited in default browser sandboxes. Standard BLE testing requires running on physical devices or compatible web-ble environments.*

### 4. Compiling & Deploying to Android
```bash
# Build the production web bundle
npm run build

# Sync web assets to the native Android platform
npx cap sync android

# Open in Android Studio to run on an emulator or physical device
npx cap open android
```
The compiled Android binary is located at the root of the project as [Hydrocheck.apk](file:///d:/Hydrocheck/Hydrocheck.apk).
