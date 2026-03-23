# GUI App Plan: Abnormal Samples Review

## 1. Overview & Objective

The goal is to provide a user-friendly interface for researchers/developers to sift through hundreds or thousands of anomalous training samples (e.g., resulting from `grad_norm_outlier` or high loss). The tool will allow users to determine if a sample is poorly formatted, contains garbage data, or is a valid edge case, and then either edit, discard, or keep the sample for future training.

## 2. Tech Stack Recommendations

Since you are working in a C++/Python environment, here are two optimal paths:

- **Option A (Desktop App):** **PyQt6 / PySide6** or **Tkinter**. Fast to build, runs locally on the dev machine, and easily interfaces with the local filesystem.
- **Option B (Web App):** **Python Flask / FastAPI backend** + **Vanilla HTML/JS/Bootstrap frontend**. Good if the review load needs to be shared among a team over a network.

*(We recommend **Option A (PyQt6)** for a dedicated locally-hosted tool that can easily serialize/deserialize JSON data on disk.)*

## 3. Core Features

1. **Data Ingestion:** Load and parse `training_sessions/abnormal_samples.json`.
2. **Filtering & Sorting:**
   - Filter by `reason` (e.g., `grad_norm_outlier`, `loss_outlier`).
   - Sort by `loss` (descending/ascending) or `grad_norm` magnitude.
3. **Data Review (The Core Loop):**
   - Present `input_text` and `target_text` clearly (with text-wrapping and scrollbars).
   - Show metrics (`loss`, `grad_norm`, `sample_id`, `epoch`).
4. **Action Assignment:**
   - **Keep:** Mark the sample as valid (false positive outlier).
   - **Edit:** Allow the reviewer to fix the `target_text` or `input_text` (e.g., if there's a typo, truncation, or tokenization error).
   - **Discard:** Mark the sample for removal from the main dataset.
5. **Export/Save:**
   - Save progress locally (in case the review takes multiple sessions).
   - Export an `action_manifest.json` or rewrite a cleaned dataset.

## 4. Proposed UI Layout
```text
+-----------------------------------------------------------------------------------+
|File: abnormal_samples.json|Total: 1542|Reviewed: 34|Pending: 1508|
+-----------------------------------------------------------------------------------+
|Filters: [Reason: All ▼]   Sort: [Grad Norm (Desc) ▼]|
+----------------------------+------------------------------------------------------+
|Outlier List|Sample Details|
|||
|[ ] ID: 5679|Norm: 23.28|Epoch: 10         Sample ID: 5679|
|[✓] ID: 5700|Norm: 22.16|Reason: grad_norm_outlier|
|[ ] ID: 5711|Norm: 20.38|Loss: 2.42        Grad Norm: 23.28|
|[ ] ID: 5682|Norm: 18.52||
|[ ] ID: 5665|Norm: 18.04|Input Text:|
|[ ] ID: 5656|Norm: 16.56|+--------------------------------------------------+|
|||Summarize: Tellson's Bank in London, sir? We||
|||have oftentimes the honour to entertain your||
|||gentlemen in||
||+--------------------------------------------------+|
|||
||Target Text:|
||+--------------------------------------------------+|
|||A||
||+--------------------------------------------------+|
|||
||Actions|
||[ Keep Sample ]  [ Discard Sample ]  [ Save Edits ]|
+----------------------------+------------------------------------------------------+
|[ Export Resolution Log ]                < Previous Sample       Next Sample >|
+-----------------------------------------------------------------------------------+
```

## 5. Implementation Phases

* **Phase 1: Backend & Data Models** ✅ COMPLETE — `tools/abnormal_review/`
  - `models.py`: `Sample`, `ActionType`, `ReviewDecision`, `ReviewState`, `SortField`
  - `data_loader.py`: `build_review_state()`, `save_progress()`, `load_progress()`, `export_resolutions()`, `export_cleaned_dataset()`
  - `__init__.py`: public package API
  - Progress checkpoint saved to `training_sessions/abnormal_review_progress.json`
  - Resolutions exported to `training_sessions/abnormal_resolutions.json`
  - Cleaned dataset exported to `training_sessions/abnormal_cleaned.json`
* **Phase 2: UI Construction** ✅ COMPLETE — tkinter (8.6) selected; PyQt6/PySide6 unavailable
  - `tools/abnormal_review/ui/list_panel.py`: `SampleListPanel` — scrollable sidebar with reason/action/sort filter controls; colour-coded rows (grey/green/red/blue per action)
  - `tools/abnormal_review/ui/detail_panel.py`: `SampleDetailPanel` — metrics bar, editable Input/Target text boxes, reviewer notes, Keep/Discard/Save Edits/Reset action buttons, Prev/Next navigation
  - `tools/abnormal_review/ui/__init__.py`: UI sub-package
* **Phase 3: Controller Logic & Actions** ✅ COMPLETE (implemented alongside Phase 2)
  - List click → `_select_sample()` populates detail pane
  - Keep / Discard / Save Edits wired to `ReviewDecision` methods; decision badge updates instantly
  - Auto-advances to next UNREVIEWED sample (200 ms delay) after each action
  - Prev/Next navigation + Ctrl+←/→ keyboard shortcuts
  - Jump-to-ID dialog via Review menu
* **Phase 4: Output Rendering** ✅ COMPLETE (implemented alongside Phase 2)
  - `tools/abnormal_review/ui/main_window.py`: `ReviewMainWindow` — full orchestration, menu bar, progress bar, status bar
  - 30-second autosave to `training_sessions/abnormal_review_progress.json`
  - File → Export Resolutions / Export Cleaned Dataset dialogs with file picker
  - `tools/abnormal_review/app.py`: CLI entry point with `--samples`, `--progress`, `--no-resume` flags
  - `review_abnormal_samples.py`: project-root launcher script

Launch the tool:

```bash
# From project root
python review_abnormal_samples.py
# or with custom paths
python review_abnormal_samples.py --samples training_sessions/abnormal_samples.json --no-resume
```
