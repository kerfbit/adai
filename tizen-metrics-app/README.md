# ADAI Training Metrics — Tizen TV App

Real-time training metrics dashboard for Samsung Smart TV (Tizen OS).  
Connects to the ADAI `metrics_api_server` over your local network and displays live training progress.

## Screenshot Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ■ ADAI  Training Metrics Dashboard   [SESSION 1 ACTIVE]   ●Connected  │
├────────────────┬────────────────────────────────┬──────────────────────┤
│ EPOCH PROGRESS │  TRAINING LOSS                 │ LEARNING RATE        │
│   2 / 10       │      2.3456                    │  0.0009500           │
│ ████░░░░ 20%  │  Running: 2.4001  Best Ep: 1   │ PERPLEXITY           │
├────────────────│                                │  10.4300             │
│ SAMPLE PROGRESS│  VALIDATION LOSS               │ GRADIENT NORM        │
│  450 / 1000    │      2.4123                    │  1.2340              │
│ ██████░░ 45%  │  Best Val: 2.3001              │ SESSION TOTALS       │
├────────────────│                                │  Total: 2,450        │
│ TIME REMAINING │  LOSS HISTORY                  │  Session ID: 1       │
│  2m 00s        │  [line chart]                  │  Time: 3m 16s        │
│ Elapsed: 3m 16s│  ─── Train  ─── Validation    │  Best Epoch: 1       │
├────────────────│                                ├──────────────────────┤
│ THROUGHPUT     │                                │ ⚙ Settings           │
│  12.5          │                                │                      │
│ samples/second │                                │ Press OK to configure│
└────────────────┴────────────────────────────────┴──────────────────────┘
│ ▲▼◄► Navigate  OK Select  BACK Back  EXIT Exit  │ Last update: 13:45:02│
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Live metrics**: loss, validation loss, learning rate, perplexity, gradient norm
- **Progress bars**: epoch and sample progress with percentage
- **Epoch loss chart**: smooth bezier curves for training and validation loss history
- **Session stats**: total samples trained, best epoch, elapsed/remaining time
- **Remote-friendly**: full D-pad navigation, no mouse/keyboard needed
- **Settings overlay**: configure API host, port, and poll interval via TV remote
- **Auto-reconnect**: retries on connection loss with visual status indicator

## Prerequisites

### On the training machine

Start the metrics API server (built from the ADAI project):

```bash
cd /path/to/adai/build
cmake .. -DBUILD_METRICS_API_SERVER=ON
make metrics_api_server
./src/metrics_api_server --port 8081
```

The server must be reachable from the TV over the local network.  
Check that port `8081` is not blocked by a firewall:

```bash
sudo ufw allow 8081/tcp   # Ubuntu/Debian
```

### On the Samsung TV

- Tizen Studio (6.x or later) with TV Extension SDK
- Samsung Smart TV in Developer Mode (Settings → Device Info → Enable Developer Mode)
- TV and development machine on the same Wi-Fi/LAN

## Project Structure

```
tizen-metrics-app/
├── config.xml          # Tizen app manifest
├── icon.png            # App icon (512×512, see note below)
├── index.html          # Dashboard HTML
├── css/
│   └── style.css       # TV-optimised styles (1920×1080)
└── js/
    ├── chart.js        # Canvas loss chart renderer
    ├── navigation.js   # D-pad / remote key handler
    └── app.js          # Metrics polling & UI update
```

## Building & Deploying

### 1. Import into Tizen Studio

1. Open Tizen Studio
2. **File → Import → Tizen → Tizen Project**
3. Select this `tizen-metrics-app/` folder
4. Choose **TV** profile, version **6.0+**

### 2. Generate Icon

Create a 512×512 PNG icon and save as `icon.png` in this folder.  
Quick placeholder using ImageMagick:

```bash
convert -size 512x512 xc:#0d1117 \
    -fill '#58a6ff' -font DejaVu-Sans-Bold -pointsize 120 \
    -gravity center -annotate 0 'ADAI' \
    icon.png
```

### 3. Configure API Host

Edit the default host/port in `js/app.js`, or use the in-app Settings at runtime:

```js
// js/app.js, top of file
var Config = {
    host: '192.168.1.100',   // ← change to your training machine IP
    port: '8081',
    ...
};
```

### 4. Package & Deploy

Use the provided `deploy.sh` script, which handles staging, signing, renaming (spaces in the WGT filename break `pkgcmd` on the TV), connecting via `sdb`, permit-to-install, and launching:

```bash
bash deploy.sh
# To install without auto-launching:
bash deploy.sh --no-launch
```

The script targets the TV at `10.0.0.10` by default. Edit `TV_SERIAL` at the top of `deploy.sh` to change the target device.

### 5. Run via Browser (testing)

Open `index.html` in Chrome/Edge with CORS disabled for local testing:

```bash
# Linux
google-chrome --disable-web-security --user-data-dir=/tmp/chrome-test \
    --app=file:///path/to/tizen-metrics-app/index.html
```

## Remote Control Mapping

| Key         | Action                        |
|-------------|-------------------------------|
| ▲ ▼ ◄ ►   | Move focus between cards      |
| OK          | Select / open settings        |
| BACK        | Close settings overlay        |
| EXIT        | Exit application              |
| Blue (F3)   | Open settings                 |
| Green (F1)  | Force refresh                 |

## Settings (In-App)

Press **Blue** or navigate to the **⚙ Settings** card and press **OK**:

- **API Host / IP** — IP address of the training machine
- **API Port** — Port of `metrics_api_server` (default: 8081)
- **Poll Interval** — How often to refresh: 1s / 2s / 5s / 10s
- **Chart Mode** — By Epoch or By Sample

Settings are persisted in `localStorage` across app restarts.

## API Endpoints Used

| Endpoint                    | Purpose                         |
|-----------------------------|---------------------------------|
| `GET /health`               | Initial connectivity check      |
| `GET /api/metrics/current`  | All real-time metrics           |
| `GET /api/session/epochs`   | Per-epoch loss history (chart)  |

See [TRAINING_METRICS_API.md](../docs/TRAINING_METRICS_API.md) for full API reference.

## Troubleshooting

| Symptom                 | Fix                                                      |
|-------------------------|----------------------------------------------------------|
| "Disconnected" status   | Check `metrics_api_server` is running; check IP/port     |
| Chart empty             | Training session not started yet; epoch data arrives after epoch 1 |
| Values show "—"         | `is_training` is false; server is idle                   |
| App won't install       | Sign the `.wgt` package with a valid Samsung certificate |
| Black screen on TV      | Check config.xml `content src` path is correct           |

## License

Same as the main ADAI project.
