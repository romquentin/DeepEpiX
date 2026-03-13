# Tuto 6: Visualize ICA Components

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">

## 1️⃣ Go to the **📈 ICA Visualization** page.
If the left sidebar is collapsed, click the <i class="bi bi-layout-sidebar-inset"></i> **Open Sidebar** button. The sidebar contains 2 tabs:

- <i class="bi bi-noise-reduction"></i> **Compute**
- <i class="bi bi-hand-index-thumb"></i> **Select**
- <i class="bi bi-grid-3x3-gap"></i> **View ICA Scalp Field Topographies**

## 2️⃣ Compute ICA with the method of your choice
In the <i class="bi bi-noise-reduction"></i> **Compute** tab:

- Enter the **Number of Components** (`n-components`).
- Choose an **ICA Method** (`fastica`, `infomax`, or `picard`).
- Set **Max Iterations** (`max-iter`).
- Set **Temporal Decimation** (`decim`).
- Click the ⚡ **Compute** button to start ICA fitting.


The ICA solution computed with `mne.preprocessing.ICA(...)` method is saved in cache as a `.fif` file. Progress and past computations appear in the **ICA History** log.

## 3️⃣ Select ICA result set your display preferences and exclude components

In the <i class="bi bi-hand-index-thumb"></i> **Select** tab:

- Pick an ICA result.
- Select ICA componets to remove from the signal.
- Enable **annotations** to display on the main graph and in the annotation overview (below the time slider).
- Set **amplitude** (1–10) to adjust signal scaling (affects how compressed or expanded the signal appears vertically).
- Pick a **color** scheme (e.g., rainbow applies group-based coloring).
- Click the 🔄 **Refresh** button on the top-left **Modebar** to apply changes.

## 4️⃣ Visualize Scalp Fiel Topographies for each component

Clicking the <i class="bi bi-grid-3x3-gap"></i> **View ICA Scalp Field Topographies** tab opens a pop-up window that displays the topographies associated with each calculated component.

## 5️⃣ Once the Graph is Displayed
**Left Modebar:**

- 🔄 **Refresh** the graph after changing ICA result, amplitude, or color settings.
- ⏩ **Navigate between pages** (the signal is displayed in 2-minute chunks for performance; this duration can be modified in config.py).
- 🧭 **Jump to previous/next event** beyond the current view (default: all selected events; can be filtered by event type).

**Right Modebar:**

- 📸 Take a **snapshot** of the current view.
- 🔍 **Zoom in** on time and components.
- 🖱️ **Pan horizontally** by clicking and dragging.
- ⏱️ **Zoom in/out** on the time axis.
- 🧼 **Autoscale** or **reset** to display the full signal duration.

**Time Range Slider:**

- Navigate through the signal timeline.
- Adjust the visible time range.

**Component Slider:**

- Use your mouse or trackpad to scroll through components vertically.

**📝 Annotation Overview**

- View annotation positions below the graph for quick reference.
