# MNS Manager GUI Operations Manual

**Binary:** `build/bin/mns_manager_gui` (portable: `build/portable/bin/mns_manager_gui`)
**Requires:** running `mns_server` daemon, Qt5 or Qt6
**Default server:** `http://localhost:8083`

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Installation](#2-installation)
3. [Launching](#3-launching)
4. [Window Layout](#4-window-layout)
   - 4.1 [Toolbar](#41-toolbar)
   - 4.2 [Models tab](#42-models-tab)
   - 4.3 [Roles tab](#43-roles-tab)
   - 4.4 [Register tab](#44-register-tab)
   - 4.5 [Detail panel](#45-detail-panel)
   - 4.6 [Actions panel](#46-actions-panel)
5. [Workflows](#5-workflows)
   - 5.1 [Connect to a server](#51-connect-to-a-server)
   - 5.2 [Register a new model](#52-register-a-new-model)
   - 5.3 [Walk a model through its lifecycle](#53-walk-a-model-through-its-lifecycle)
   - 5.4 [Promote a candidate to production](#54-promote-a-candidate-to-production)
   - 5.5 [Inspect a model](#55-inspect-a-model)
   - 5.6 [Retire and delete a model](#56-retire-and-delete-a-model)
   - 5.7 [Filter models by state or role](#57-filter-models-by-state-or-role)
6. [Color Coding](#6-color-coding)
7. [Keyboard and Mouse Reference](#7-keyboard-and-mouse-reference)
8. [Relationship to mns_cli](#8-relationship-to-mns_cli)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Purpose

The MNS Manager GUI provides a graphical interface for the ADAI Model Name Service. It is the visual counterpart to `mns_cli` and is intended for operators who prefer point-and-click model management over command-line workflows.

The GUI connects to the same `mns_server` HTTP API used by all other ADAI components. It can run from any machine that can reach the MNS port.

**Capabilities:**

- Browse and filter registered models by state or role.
- View full model records including architecture, artifact location, and training history.
- Register new models with architecture parameters and tags.
- Drive lifecycle state transitions: initializing, training, candidate, production, retired.
- Promote candidates to production for a role.
- Retire and hard-delete models (with confirmation dialogs).
- View role-to-production-model mappings.
- Connect to any MNS server at runtime by editing the URL in the toolbar.

---

## 2. Installation

### Build prerequisites

- Qt5 (`qtbase5-dev`) or Qt6 (`qt6-base-dev`)
- cpp-httplib
- SQLite3 (`libsqlite3-dev`)

### Build

```bash
cmake --preset portable
cmake --build --preset portable --target mns_manager_gui
```

The binary is placed at `build/portable/bin/mns_manager_gui`.

The server bundle install script (`scripts/install_server_bundle.sh`) does not install the GUI by default since it is a desktop application, not a server daemon. Copy the binary manually to any workstation that needs it:

```bash
scp build/portable/bin/mns_manager_gui user@workstation:/usr/local/bin/
```

### Qt library note

If you encounter Qt plugin errors on snap-based systems, set the plugin path before launching:

```bash
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins
mns_manager_gui
```

---

## 3. Launching

```
mns_manager_gui [--url URL] [--config PATH] [--help]
```

| Option | Description |
|--------|-------------|
| `--url URL` | MNS server URL (overrides config) |
| `--config PATH` | Path to `config.conf` for `NAME_SERVICE_URL` fallback |
| `--help` | Print usage and exit |

**Server URL resolution order:** `--url` flag > `NAME_SERVICE_URL` in config/env > `http://localhost:8083`.

**Examples:**

```bash
# Connect to local server (default)
mns_manager_gui

# Connect to a remote server
mns_manager_gui --url http://192.168.1.19:8083

# Use a specific config file
mns_manager_gui --config /etc/adai/config.conf
```

On startup the GUI immediately attempts a health check against the configured server. The toolbar status indicator shows **Connected** (green) or **Disconnected** (red).

---

## 4. Window Layout

The window is divided into two resizable halves separated by a draggable splitter.

```
+--[ Toolbar: Server URL | Connect | Refresh | Status ]--+
|                           |                             |
|   [ Models | Roles |      |   Model Detail              |
|     Register ]            |   (pretty-printed JSON)     |
|                           |                             |
|   (tab content: table     |   +-----------------------+ |
|    or registration form)  |   | Actions               | |
|                           |   |  Set Training          | |
|                           |   |  Set Candidate         | |
|                           |   |  Promote               | |
|                           |   |  Retire | Delete       | |
|                           |   +-----------------------+ |
+---------------------------------------------------------+
```

### 4.1 Toolbar

The toolbar spans the top of the window.

| Element | Description |
|---------|-------------|
| **Server** field | Editable URL. Change it and click **Connect** to switch servers at runtime. |
| **Connect** button | Runs a health check against the URL in the field. On success, refreshes all tables. |
| **Refresh** button | Re-fetches models and roles from the current server without changing the URL. |
| **Status** label | Shows **Connected** (green) or **Disconnected** (red) and recent action results (e.g. "Registered: my-model"). |

### 4.2 Models tab

A table of all registered models. Each row shows:

| Column | Content |
|--------|---------|
| Name | Model name (e.g. `adai-chatbot-v3`) |
| Role | Assigned role (e.g. `chatbot`) |
| State | Lifecycle state, color-coded (see [Color Coding](#6-color-coding)) |
| Model ID | Truncated UUID (`550e8400...`) |
| Updated | Last state-change timestamp (UTC) |

**Filters** above the table narrow the list:

- **State** dropdown: `(all)`, `initializing`, `training`, `candidate`, `production`, `retired`. Selecting a state re-fetches the model list with a server-side filter.
- **Role** dropdown: `(all)` plus an editable field for typing a role name. Selecting a role re-fetches with a server-side filter.

Filters combine: selecting State=`candidate` and Role=`chatbot` shows only candidate chatbot models.

**Clicking a row** selects the model, loads its full record into the detail panel, and pre-fills the action fields (model name, role) for convenience.

### 4.3 Roles tab

A two-column table showing all known roles.

| Column | Content |
|--------|---------|
| Role | Role name |
| Production Model | Name of the model currently in production for this role, or `(none)` |

**Clicking a row** pre-fills the Promote action fields (role and model name) and loads the production model's detail if one exists.

### 4.4 Register tab

A form for registering a new model in the MNS.

| Field | Required | Description |
|-------|----------|-------------|
| Model Name | Yes | Lowercase, hyphens, digits. E.g. `adai-chatbot-v3` |
| Role | Yes | Logical role. E.g. `chatbot`, `reward-model` |
| d_model | No | Model dimension (default: 128) |
| num_heads | No | Attention heads (default: 4) |
| d_ff | No | Feed-forward dimension (default: 512) |
| encoder_layers | No | Encoder layer count (default: 6) |
| decoder_layers | No | Decoder layer count (default: 6) |
| max_seq_length | No | Maximum sequence length (default: 1024) |
| Tags | No | Comma-separated `key=value` pairs. E.g. `owner=rodney, dataset=minipile` |

Click **Register Model** to submit. On success, the models table refreshes and the status bar shows the registered name. On failure (e.g. duplicate name), a dialog shows the server error.

Architecture defaults in the spin boxes are hard-coded. For config-driven defaults, use `mns_cli register` which reads `config.conf`.

### 4.5 Detail panel

The upper-right area shows the full JSON record of the currently selected model, pretty-printed with indentation for readability. The view is read-only.

The detail includes:

- Identity: `model_id`, `model_name`, `role`, `state`
- Timestamps: `created_utc`, `updated_utc`
- Artifact location: `host`, `path`, `checksum`, `format`
- Architecture: `d_model`, `num_heads`, `d_ff`, layer counts, `max_seq_length`
- Training history: array of `{run_id, epochs, final_loss, started_utc, finished_utc}`
- Tags: arbitrary key-value pairs

The detail refreshes automatically whenever you click a model row or complete a state transition.

### 4.6 Actions panel

The lower-right area under the detail view. All actions operate on the **currently selected model** (shown in the detail panel). Actions that require a model show a warning dialog if nothing is selected.

#### Set Training

| Field | Description |
|-------|-------------|
| Run ID | Unique training run identifier (required) |

Transitions the selected model to `training` state. Valid from `initializing` or `candidate`.

#### Set Candidate

| Field | Description |
|-------|-------------|
| Artifact | Absolute path to weight file (optional but recommended) |

Transitions the selected model to `candidate` state. Uses the Run ID from the field above if present. Valid from `training`, `initializing`, or `retired`.

#### Promote

| Field | Description |
|-------|-------------|
| Role | Target role for promotion |
| Model | Model name to promote |

Both fields are pre-filled when you click a model or role row. Promotes a `candidate` model to `production` for the specified role. If another model already holds production for that role, it is automatically retired.

#### Retire

Transitions the selected model to `retired`. Shows a confirmation dialog. Valid from any state except `retired`.

#### Delete

Hard-deletes the selected model record from the database. Shows a confirmation dialog. Only permitted for models in `initializing` or `retired` state. Does **not** delete weight files from disk.

---

## 5. Workflows

### 5.1 Connect to a server

1. Edit the **Server** field in the toolbar to the desired URL (e.g. `http://192.168.1.19:8083`).
2. Click **Connect**.
3. The status indicator turns green if the server responds. Both the Models and Roles tables are populated automatically.

You can switch servers at any time by editing the URL and clicking Connect again.

### 5.2 Register a new model

1. Click the **Register** tab.
2. Enter a model name (e.g. `adai-chatbot-v4`) and role (e.g. `chatbot`).
3. Adjust architecture parameters if they differ from defaults.
4. Optionally add tags (e.g. `owner=rodney, dataset=minipile-v2`).
5. Click **Register Model**.
6. The Models table refreshes. The new model appears with state `initializing`.

### 5.3 Walk a model through its lifecycle

1. In the **Models** tab, click the model row to select it.
2. In the **Actions** panel, enter a **Run ID** (e.g. `run-42`).
3. Click **Set Training**. The detail panel updates to show `state: training`.
4. After training completes, enter the checkpoint path in **Artifact** (e.g. `/opt/adai/training_sessions/session_3_best.bin`).
5. Click **Set Candidate**. The detail panel updates to show `state: candidate` with the artifact attached.

Steps 3 and 5 happen automatically when `incremental_trainer` has `NAME_SERVICE_URL` and `MODEL_NAME` configured. The GUI is useful for manual imports, testing, or correcting state when automation fails.

### 5.4 Promote a candidate to production

1. Select the candidate model in the **Models** tab (or click the role row in the **Roles** tab).
2. The **Role** and **Model** fields in the Promote row are pre-filled.
3. Verify both fields, then click **Promote**.
4. The status bar shows the result. Both the Models and Roles tables refresh. The previously promoted model (if any) is automatically retired.

### 5.5 Inspect a model

1. Click any model row in the **Models** tab.
2. The **Detail** panel shows the full JSON record with indented formatting.
3. Scroll to find architecture, artifact location, training history, and tags.

Alternatively, click a role row in the **Roles** tab to jump directly to its production model's detail.

### 5.6 Retire and delete a model

1. Select the model in the **Models** tab.
2. Click **Retire**. Confirm the dialog. The model moves to `retired` state.
3. Click **Delete**. Confirm the dialog. The model record is removed from the database.

Delete is only enabled for `initializing` and `retired` models. If you try to delete a `training`, `candidate`, or `production` model, the server returns an error and a dialog explains the constraint.

Weight files are never deleted by the GUI or the MNS server. Remove them from disk manually after confirming they are no longer needed.

### 5.7 Filter models by state or role

1. In the **Models** tab, use the **State** dropdown to select a lifecycle state (e.g. `production`).
2. Use the **Role** dropdown (or type a role name) to narrow by role.
3. The table re-fetches from the server with the selected filters applied server-side.
4. Select `(all)` in either dropdown to remove that filter.

---

## 6. Color Coding

Model states in the Models table are color-coded for quick scanning:

| State | Color | Meaning |
|-------|-------|---------|
| `initializing` | Black (default) | Registered but never trained |
| `training` | Blue | Training run in progress |
| `candidate` | Orange | Training complete, awaiting promotion |
| `production` | Green | Active production model for its role |
| `retired` | Gray | Superseded or manually retired |

---

## 7. Keyboard and Mouse Reference

| Action | How |
|--------|-----|
| Select a model | Click any cell in its row |
| Switch tabs | Click the tab header (Models, Roles, Register) |
| Resize panels | Drag the vertical splitter between left and right halves |
| Submit registration | Click **Register Model** (no keyboard shortcut) |
| Refresh data | Click **Refresh** in the toolbar |
| Reconnect | Edit URL, click **Connect** |

---

## 8. Relationship to mns_cli

The GUI and `mns_cli` are interchangeable interfaces to the same MNS server API. Every action available in the GUI has a `mns_cli` equivalent:

| GUI action | mns_cli equivalent |
|------------|--------------------|
| Models tab (list) | `mns_cli list [--state S] [--role R]` |
| Click model row | `mns_cli get <name>` |
| Register tab | `mns_cli register <name> <role> [opts]` |
| Set Training button | `mns_cli set-training <name> <run-id>` |
| Set Candidate button | `mns_cli set-candidate <name> <run-id> --artifact-path PATH` |
| Promote button | `mns_cli promote <role> <model-name>` |
| Retire button | `mns_cli set-candidate <name> <run-id>` then set state to retired via API |
| Delete button | `mns_cli delete <name>` |
| Roles tab | `mns_cli roles` |
| Roles row click | `mns_cli resolve-role <role>` |
| Connect / Refresh | `mns_cli health` |

Use `mns_cli` for scripting, automation, and headless servers. Use the GUI for interactive exploration, bulk inspection, and operators who prefer visual feedback.

---

## 9. Troubleshooting

### Status shows "Disconnected" on startup

The GUI could not reach the MNS server at the configured URL.

- Verify the server is running: `systemctl status adai-mns`
- Verify the port is listening: `ss -tlnp | grep 8083`
- If the server is on a remote host, check firewall rules and use `--url http://<host>:8083`.
- Edit the URL in the toolbar and click **Connect** to retry.

### "Registration Failed" dialog

The server rejected the registration request. Common causes:

| Error message | Fix |
|---------------|-----|
| `model_name already registered` | Choose a different name or delete the existing model first. |
| `invalid model_name` | Names must be lowercase alphanumeric with hyphens. No uppercase, spaces, or special characters. |
| `model_name required` | The name field is empty. |

### "State Transition Failed" dialog

The server rejected the state change. This means the current state does not allow the requested transition. See the lifecycle diagram in the [mns_cli guide](mns-cli-guide.md#5-model-lifecycle) for the valid state machine.

Common situations:

| Want to do | Current state | Fix |
|-----------|--------------|-----|
| Set Training on a production model | `production` | Promote a different model first, or register a new name. |
| Delete a candidate | `candidate` | Retire it first, then delete. |
| Promote a training model | `training` | Complete training with Set Candidate first. |

### "Promotion Failed" dialog

| Error message | Fix |
|---------------|-----|
| `model must be in candidate state` | The model needs to be in `candidate` state before promotion. Use Set Candidate first. |
| `model not found` | Check the model name in the Promote field matches exactly. |

### Detail panel is empty after clicking a row

The server returned an empty response. Click **Refresh** to re-fetch the models table. If the problem persists, check the server logs: `journalctl -u adai-mns -n 20`.

### Tables do not update after an action

All mutation actions (register, state transition, promote, retire, delete) automatically trigger a table refresh. If the table appears stale, click **Refresh** in the toolbar. Verify the action actually succeeded by checking the status bar message.

### Window does not appear (headless/SSH)

The GUI requires a display server (X11 or Wayland). Over SSH, use X forwarding:

```bash
ssh -X user@server
mns_manager_gui
```

For headless management, use `mns_cli` instead.
