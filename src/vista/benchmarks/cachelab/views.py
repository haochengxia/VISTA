"""Render analysis results as fixed-width PNG views for the agent.

Every image is exactly 1024 pixels wide, so `inspect` and `read_pixels`
regions map directly onto it. Encoding rules (validated with the dataviz
palette checker):

- queue identity uses the categorical slots in a fixed order, resident queues
  first, then ghosts (baseline: S blue, M orange, G aqua), in every view;
- magnitude (miss ratio) is a single blue ramp; a change against a baseline is
  diverging blue (better) <-> red (worse) around a gray midpoint;
- hit and miss marks differ by shape in ink, so color stays with the queue;
- no panel has two y-scales; different units get their own panel.

The exact numbers behind each view are returned as text next to the image.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap,
    Normalize,
    TwoSlopeNorm,
    to_hex,
)
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

WIDTH = 1024
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL = "#f0efec"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
BETTER = "#2a78d6"
WORSE = "#e34948"
OTHER = "#c3c2b7"
EXTRA = "#6250d6"  # bypassed requests in the miss-cause stack
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SEQUENTIAL = LinearSegmentedColormap.from_list("cachelab-blue", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "cachelab-diverging", ("#184f95", "#3987e5", "#9ec5f4", NEUTRAL, "#f2a3a2", "#e34948", "#a32a2a")
)
LINE = 1.45  # ~2px at 100 dpi


def queue_colors(queues: list[str], ghosts: list[str]) -> dict[str, str]:
    names = list(queues) + list(ghosts)
    if len(queues) <= 4:
        return {name: CATEGORICAL[index % len(CATEGORICAL)] for index, name in enumerate(names)}
    # Many queues are ordered levels (MQ's Q0..Q7): one hue, light to dark.
    ramp = LinearSegmentedColormap.from_list("levels", BLUE_RAMP[1:])
    colors = {name: to_hex(ramp(index / (len(queues) - 1))) for index, name in enumerate(queues)}
    for index, name in enumerate(ghosts):
        colors[name] = CATEGORICAL[2 + index]
    return colors


def _figure(height: int) -> Figure:
    figure = Figure(figsize=(WIDTH / 100, height / 100), dpi=100, facecolor=SURFACE)
    FigureCanvasAgg(figure)
    return figure


def _style(axis, *, ylabel: str | None = None, grid: bool = True) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BASELINE)
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=9, length=3, width=0.6)
    if grid:
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    if ylabel:
        axis.set_ylabel(ylabel, color=SECONDARY, fontsize=9)


def _handover_marks(axes, marks: list[int], start: int, end: int) -> None:
    """A dashed line where a policy took over the live cache, on every panel."""
    shown = [mark for mark in marks if start <= mark < end]
    for index, axis in enumerate(axes):
        for mark in shown:
            axis.axvline(mark, color=INK, linewidth=0.8, linestyle=(0, (4, 3)), zorder=5)
            if index == 0:
                axis.annotate("handover", (mark, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2), textcoords="offset points", fontsize=8, color=INK, va="top")


def _requests_axis(axis, start: int, end: int, *, label: bool = True) -> None:
    axis.set_xlim(start, end)
    span = end - start
    scale, unit = (1000, "k") if span >= 5000 else (1, "")
    axis.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _: f"{value / scale:g}{unit}")
    )
    if label:
        axis.set_xlabel("request (virtual time)", color=SECONDARY, fontsize=9)


def _title(figure: Figure, title: str, subtitle: str = "") -> None:
    figure.text(0.012, 0.995, title, ha="left", va="top", fontsize=12, color=INK, weight="bold")
    if subtitle:
        figure.text(0.012, 0.995 - 22 / (figure.get_figheight() * 100), subtitle, ha="left", va="top", fontsize=9, color=SECONDARY)


def _png(figure: Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, facecolor=SURFACE)
    with Image.open(io.BytesIO(buffer.getvalue())) as image:
        rgb = image.convert("RGB")
        if rgb.width != WIDTH:
            raise RuntimeError("A view must be exactly 1024 pixels wide")
        out = io.BytesIO()
        rgb.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _centers(bounds: np.ndarray) -> np.ndarray:
    return (bounds[:-1] + bounds[1:]) / 2


def _label_end(axis, x: float, y: float, text: str, color: str = SECONDARY) -> None:
    if np.isfinite(y):
        axis.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=color, annotation_clip=False)


def _label_ends(axis, items: list[tuple[float, float, str, str]], spacing: float = 11.0) -> None:
    """Direct labels at line ends, nudged apart vertically so none overlap."""
    items = [item for item in items if np.isfinite(item[1])]
    if not items:
        return
    axis.figure.canvas.draw_idle()
    points = [(axis.transData.transform((x, y))[1], x, y, text, color) for x, y, text, color in items]
    points.sort()
    placed: list[float] = []
    for pixel, *_ in points:
        placed.append(max(pixel, placed[-1] + spacing) if placed else pixel)
    for (pixel, x, y, text, color), target in zip(points, placed, strict=True):
        axis.annotate(text, (x, y), xytext=(4, (target - pixel) * 72 / 100), textcoords="offset points", va="center", fontsize=8.5, color=color, annotation_clip=False)


# -- overview -------------------------------------------------------------------


def render_overview(panels: list[dict[str, Any]], *, title: str, subtitle: str, delta: bool) -> bytes:
    """One heatmap per trace: rows are cache sizes, columns are time windows."""
    rows = len(panels)
    header = 104
    height = min(1024, header + 4 + 228 * rows)
    figure = _figure(height)
    _title(figure, title, subtitle)
    top = 1 - header / height
    bottom = 46 / height
    gap = 34 / height
    panel_height = (top - bottom - gap * (rows - 1)) / rows
    if delta:
        bound = max(1e-4, max(float(np.nanmax(np.abs(p["matrix"]))) for p in panels))
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
        cmap = DIVERGING
    else:
        vmax = max(float(np.nanmax(p["matrix"])) for p in panels)
        norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-3))
        cmap = SEQUENTIAL
    image = None
    for position, panel in enumerate(panels):
        y0 = top - (position + 1) * panel_height - position * gap
        axis = figure.add_axes((0.10, y0, 0.70, panel_height))
        _style(axis, grid=False)
        matrix = panel["matrix"]
        bounds = panel["bounds"]
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=(bounds[0], bounds[-1], matrix.shape[0] - 0.5, -0.5),
        )
        for row in range(1, matrix.shape[0]):
            axis.axhline(row - 0.5, color=SURFACE, linewidth=1.5)
        axis.set_yticks(range(matrix.shape[0]))
        axis.set_yticklabels([f"C={c:,}" for c in panel["capacities"]], fontsize=8.5)
        _requests_axis(axis, int(bounds[0]), int(bounds[-1]), label=position == rows - 1)
        axis.set_title(panel["label"], loc="left", fontsize=10, color=INK, pad=4)
        for row, text in enumerate(panel["row_labels"]):
            axis.annotate(text, (1.0, row), xycoords=("axes fraction", "data"), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK, annotation_clip=False)
    colorbar_axis = figure.add_axes((0.10, 1 - 64 / height, 0.30, 10 / height))
    bar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=SECONDARY, length=2)
    legend = (
        "miss-ratio change vs baseline per window (blue = fewer misses, red = more)"
        if delta
        else "miss ratio per window (darker = more misses)"
    )
    figure.text(0.42, 1 - 59 / height, legend, fontsize=8.5, color=SECONDARY, va="center")
    figure.text(0.81, 1 - header / height + 20 / height, "whole trace", fontsize=8.5, color=SECONDARY, va="bottom")
    return _png(figure)


# -- queue replay ---------------------------------------------------------------


def render_queues(profile: dict[str, Any], *, title: str, subtitle: str, marks: list[int] = ()) -> bytes:
    height = 1024
    figure = _figure(height)
    _title(figure, title, subtitle)
    bounds = profile["bounds"]
    x = _centers(bounds)
    start, end = int(bounds[0]), int(bounds[-1])
    queues, ghosts = profile["queues"], profile["ghosts"]
    colors = queue_colors(queues, ghosts)
    axes = figure.subplots(5, 1, sharex=True, gridspec_kw={"hspace": 0.42, "left": 0.08, "right": 0.78, "top": 0.92, "bottom": 0.06})

    axis = axes[0]
    _style(axis, ylabel="objects")
    axis.set_title("Queue occupancy", loc="left", fontsize=10, color=INK, pad=3)
    ends = []
    for name in queues + ghosts:
        values = profile["occupancy"][name]
        axis.plot(x, values, color=colors[name], linewidth=LINE, label=name)
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size:
            ends.append((x[finite[-1]], values[finite[-1]], name, colors[name]))
    axis.set_ylim(bottom=0)
    _label_ends(axis, ends)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5, labelcolor=SECONDARY)

    axis = axes[1]
    _style(axis, ylabel="share of requests")
    axis.set_title("Misses by cause (stacked): cold, or re-requested after eviction from a queue", loc="left", fontsize=10, color=INK, pad=3)
    breakdown = profile["miss_breakdown"]
    labels = list(breakdown)
    stack_colors = []
    for label in labels:
        if label == "cold":
            stack_colors.append(OTHER)
        elif label == "bypassed":
            stack_colors.append(EXTRA)
        else:
            stack_colors.append(colors[label.removeprefix("after ").removesuffix(" eviction")])
    axis.stackplot(x, [breakdown[label] for label in labels], colors=stack_colors, labels=labels, edgecolor=SURFACE, linewidth=0.8)
    axis.plot(x, profile["miss_ratio"], color=INK, linewidth=0.9, label="total miss ratio")
    axis.set_ylim(0, max(0.05, float(np.nanmax(profile["miss_ratio"])) * 1.08))
    axis.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=8, labelcolor=SECONDARY)

    axis = axes[2]
    _style(axis, ylabel="per 1k requests")
    axis.set_title("Evictions by queue (shaded: evicted without a single hit)", loc="left", fontsize=10, color=INK, pad=3)
    ends = []
    for name in queues:
        total = profile["evictions"][name] * 1000
        unused = profile["zero_hit_evictions"][name] * 1000
        axis.fill_between(x, 0, unused, color=colors[name], alpha=0.22, linewidth=0)
        axis.plot(x, total, color=colors[name], linewidth=LINE, label=name)
        ends.append((x[-1], total[-1], name, colors[name]))
    axis.set_ylim(bottom=0)
    _label_ends(axis, ends)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5, labelcolor=SECONDARY)

    axis = axes[3]
    _style(axis, ylabel="per 1k requests")
    axis.set_title("Transfers between queues, and ghost hits (colored by source)", loc="left", fontsize=10, color=INK, pad=3)
    plotted = False
    for label, values in profile["moves"].items():
        source, target = label.split("->")
        if source == target:
            continue
        axis.plot(x, values * 1000, color=colors[source], linewidth=LINE, label=f"{label} move")
        plotted = True
    for name, values in profile["ghost_hits"].items():
        axis.plot(x, values * 1000, color=colors[name], linewidth=LINE, label=f"{name} hit -> admit")
        plotted = True
    axis.set_ylim(bottom=0)
    if plotted:
        axis.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=8, labelcolor=SECONDARY)

    axis = axes[4]
    _style(axis, ylabel="per 1k requests")
    axis.set_title("Reinsertions within a queue (second chances at the tail)", loc="left", fontsize=10, color=INK, pad=3)
    plotted = False
    for label, values in profile["moves"].items():
        source, target = label.split("->")
        if source != target:
            continue
        axis.plot(x, values * 1000, color=colors[source], linewidth=LINE, label=f"{source} reinsert")
        plotted = True
    axis.set_ylim(bottom=0)
    if plotted:
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5, labelcolor=SECONDARY)
    _requests_axis(axis, start, end)
    _handover_marks(axes, list(marks), start, end)
    return _png(figure)


# -- object timelines --------------------------------------------------------------


def render_objects(timelines: list[dict[str, Any]], *, queues: list[str], ghosts: list[str], start: int, end: int, title: str, subtitle: str) -> bytes:
    rows = max(1, len(timelines))
    height = int(min(1024, 150 + 46 * rows))
    figure = _figure(height)
    _title(figure, title, subtitle)
    axis = figure.add_axes((0.17, 62 / height, 0.80, 1 - 150 / height))
    _style(axis, grid=False)
    colors = queue_colors(queues, ghosts)
    for row, line in enumerate(timelines):
        y = row
        axis.axhline(y, color=GRID, linewidth=0.6, zorder=0)
        for a, b, ghost, _ in line["ghost"]:
            axis.broken_barh([(a, max(b - a, 1))], (y - 0.07, 0.14), facecolors=colors.get(ghost, OTHER), alpha=0.55, zorder=1)
        for a, b, name in line["residency"]:
            axis.broken_barh([(a, max(b - a, 1))], (y - 0.22, 0.44), facecolors=colors.get(name, OTHER), zorder=2)
        hits = [t for t, outcome, _ in line["requests"] if outcome == 0]
        misses = [t for t, outcome, _ in line["requests"] if outcome in (1, 2)]
        bypassed = [t for t, outcome, _ in line["requests"] if outcome == 3]
        axis.scatter(hits, [y] * len(hits), s=16, color=INK, edgecolors=SURFACE, linewidths=0.8, zorder=4)
        axis.scatter(misses, [y] * len(misses), s=34, marker="x", color=INK, linewidths=1.3, zorder=4)
        axis.scatter(bypassed, [y] * len(bypassed), s=26, marker="v", color=INK, zorder=4)
        for t, _, _ in line["evictions"]:
            axis.vlines(t, y - 0.34, y + 0.34, color=INK, linewidth=1.2, zorder=3)
    axis.set_ylim(rows - 0.5, -0.5)
    axis.set_yticks(range(rows))
    axis.set_yticklabels(
        [f"#{row} …{line['key'] & 0xFFFFFF:06x}  ({line.get('requests_total', 0)} req)" for row, line in enumerate(timelines)],
        fontsize=8.5,
    )
    _requests_axis(axis, start, end)
    handles = [matplotlib.patches.Patch(color=colors[name], label=f"resident in {name}") for name in queues]
    handles += [matplotlib.patches.Patch(color=colors[name], alpha=0.55, label=f"key in ghost {name}") for name in ghosts]
    handles += [
        matplotlib.lines.Line2D([], [], marker="o", linestyle="", color=INK, markersize=5, label="hit"),
        matplotlib.lines.Line2D([], [], marker="x", linestyle="", color=INK, markersize=6, label="miss"),
        matplotlib.lines.Line2D([], [], marker="|", linestyle="", color=INK, markersize=10, label="eviction"),
    ]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.17, 1 - 40 / height), ncol=min(len(handles), 7), frameon=False, fontsize=8.5, labelcolor=SECONDARY, handlelength=1.4, columnspacing=1.2)
    return _png(figure)


# -- A/B diff --------------------------------------------------------------------


def render_diff(diff: dict[str, Any], *, base_name: str, cand_name: str, title: str, subtitle: str, marks: list[int] = ()) -> bytes:
    height = 800
    figure = _figure(height)
    _title(figure, title, subtitle)
    bounds = diff["bounds"]
    x = _centers(bounds)
    start, end = int(bounds[0]), int(bounds[-1])
    axes = figure.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0.38, "left": 0.08, "right": 0.88, "top": 0.90, "bottom": 0.08, "height_ratios": (1.2, 1.0, 0.9)})

    axis = axes[0]
    _style(axis, ylabel="miss ratio")
    axis.set_title("Miss ratio per window", loc="left", fontsize=10, color=INK, pad=3)
    axis.plot(x, diff["base_miss"], color=MUTED, linewidth=LINE, label=base_name)
    axis.plot(x, diff["cand_miss"], color=BETTER, linewidth=LINE, label=cand_name)
    _label_ends(axis, [(x[-1], diff["base_miss"][-1], base_name, SECONDARY), (x[-1], diff["cand_miss"][-1], cand_name, INK)])
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5, labelcolor=SECONDARY)

    axis = axes[1]
    _style(axis, ylabel="per 1k requests")
    axis.set_title(f"Requests whose outcome differs: up = only {cand_name} hits, down = only {base_name} hits", loc="left", fontsize=10, color=INK, pad=3)
    width = np.diff(bounds) * 0.86
    axis.bar(x, diff["improvements"] * 1000, width=width, color=BETTER, label=f"{cand_name} better")
    axis.bar(x, -diff["regressions"] * 1000, width=width, color=WORSE, label=f"{cand_name} worse")
    axis.axhline(0, color=BASELINE, linewidth=0.8)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5, labelcolor=SECONDARY)

    axis = axes[2]
    _style(axis, ylabel="hits gained")
    axis.set_title(f"Cumulative hits gained by {cand_name} over {base_name}", loc="left", fontsize=10, color=INK, pad=3)
    net = np.cumsum((diff["improvements"] - diff["regressions"]) * np.diff(bounds))
    axis.plot(x, net, color=INK, linewidth=LINE)
    axis.axhline(0, color=BASELINE, linewidth=0.8)
    _label_end(axis, x[-1], net[-1], f"{net[-1]:+,.0f}", INK)
    _requests_axis(axis, start, end)
    _handover_marks(axes, list(marks), start, end)
    return _png(figure)


def render_dashboard(rows: list[dict[str, Any]], columns: list[str], *, latest: str, best: str | None, title: str, subtitle: str) -> bytes:
    """Every experiment against S3-FIFO per cell (heatmap) and the search's progress (line)."""
    count = len(rows)
    heat_height = min(560, 40 + 22 * count)
    height = 140 + heat_height + 300
    figure = _figure(height)
    _title(figure, title, subtitle)
    top = 1 - 96 / height
    heat_bottom = top - heat_height / height
    axis = figure.add_axes((0.17, heat_bottom, 0.70, top - heat_bottom))
    matrix = np.array([row["cells"] for row in rows], dtype=float) * 100
    # The scale follows the agent's own experiments; far-off references saturate.
    own = matrix[[i for i, row in enumerate(rows) if not row["reference"]]] if any(not row["reference"] for row in rows) else matrix
    finite = np.abs(own[np.isfinite(own)])
    bound = max(0.5, float(np.quantile(finite, 0.95))) if finite.size else 1.0
    image = axis.imshow(matrix, aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound), interpolation="nearest")
    axis.set_yticks(range(count))
    labels = []
    for row in rows:
        mark = " ★" if row["id"] == best else (" ◀" if row["id"] == latest else "")
        labels.append(f"{row['id']}{mark}  {100 * row['mean']:+.2f}%")
    axis.set_yticklabels(labels, fontsize=8.5, color=SECONDARY)
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels(columns, fontsize=7.5 if len(columns) > 8 else 8.5, color=SECONDARY, rotation=30 if len(columns) > 6 else 0, ha="right" if len(columns) > 6 else "center")
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    if len(columns) <= 12 and count <= 24:
        for i in range(count):
            for j in range(len(columns)):
                value = matrix[i, j]
                if np.isfinite(value):
                    axis.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7.5, color=INK if abs(value) < 0.6 * bound else SURFACE)
    references = [i for i, row in enumerate(rows) if row["reference"]]
    if references and len(references) < count:
        axis.axhline(max(references) + 0.5, color=INK, linewidth=0.8)
    axis.set_title("Change vs S3-FIFO per cell, % (rows above the line: references; ★ best so far, ◀ latest)", loc="left", fontsize=10, color=INK, pad=4)
    bar = figure.add_axes((0.89, heat_bottom, 0.015, top - heat_bottom))
    figure.colorbar(image, cax=bar).ax.tick_params(labelsize=8, colors=SECONDARY)

    axis = figure.add_axes((0.08, 0.07, 0.78, 230 / height))
    _style(axis, ylabel="mean change vs S3-FIFO, %")
    ours = [row for row in rows if not row["reference"]]
    xs = np.arange(1, len(ours) + 1)
    means = np.array([100 * row["mean"] for row in ours])
    if len(ours):
        axis.plot(xs, means, "o", color=MUTED, markersize=4)
        axis.step(xs, np.minimum.accumulate(means), where="post", color=BETTER, linewidth=LINE)
        _label_end(axis, xs[-1], float(np.minimum.accumulate(means)[-1]), "best so far", BETTER)
        axis.set_xlim(0.5, max(len(ours), 2) + 0.5)
        axis.set_xticks(xs)
        axis.set_xticklabels([row["id"] for row in ours], fontsize=8, color=SECONDARY)
    axis.axhline(0, color=INK, linewidth=0.9)
    # The axis follows the agent's experiments; references far outside it are listed, not drawn.
    finite = means[np.isfinite(means)] if len(ours) else np.array([])
    low, high = (min(0.0, float(finite.min())), max(0.0, float(finite.max()))) if finite.size else (-1.0, 1.0)
    pad = max(0.25, 0.15 * (high - low))
    low, high = low - pad, high + pad
    labels = [(axis.get_xlim()[1], 0.0, "s3fifo", SECONDARY)]
    outside = []
    for row in rows:
        if not row["reference"] or not np.isfinite(row["mean"]):
            continue
        value = 100 * row["mean"]
        if low <= value <= high:
            axis.axhline(value, color=BASELINE, linewidth=0.8, linestyle=(0, (4, 3)))
            labels.append((axis.get_xlim()[1], value, row["id"], MUTED))
        else:
            outside.append(f"{row['id']} {value:+.1f}%")
    axis.set_ylim(low, high)
    _label_ends(axis, labels)
    if outside:
        axis.text(0.01, 0.97, "off scale: " + ", ".join(outside), transform=axis.transAxes, fontsize=8, color=MUTED, va="top")
    axis.set_title("Mean change by experiment (dots) and the best so far (line); dashed: references", loc="left", fontsize=10, color=INK, pad=3)
    return _png(figure)


def render_knobs(profile: dict[str, Any], *, title: str, subtitle: str) -> bytes:
    """One heatmap per knob: values (rows) x windows (columns) plus the whole range,
    colored by the change in misses vs S3-FIFO; a dot marks each window's best value."""
    knobs = profile["knobs"]
    rows = sum(len(data["values"]) for data in knobs.values())
    row_px, gap_px = 24, 64
    height = 100 + rows * row_px + (gap_px - 20) * len(knobs) + 44
    figure = _figure(height)
    _title(figure, title, subtitle)
    matrices = [np.column_stack([data["relative"], data["overall"]]) * 100 for data in knobs.values()]
    finite = np.abs(np.concatenate([m.ravel() for m in matrices]))
    finite = finite[np.isfinite(finite)]
    bound = max(0.5, float(np.quantile(finite, 0.95))) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    bounds = profile["bounds"]
    windows = len(bounds) - 1
    top = 1 - 100 / height
    image = None
    for (knob, data), matrix in zip(knobs.items(), matrices, strict=True):
        count = len(data["values"])
        panel = count * row_px / height
        bottom = top - (gap_px - 20) / height - panel
        axis = figure.add_axes((0.15, bottom, 0.72, panel))
        image = axis.imshow(matrix, aspect="auto", cmap=DIVERGING, norm=norm, interpolation="nearest")
        axis.axvline(windows - 0.5, color=SURFACE, linewidth=4)
        best = [data["values"].index(label) for label in data["best"]]
        default = data["values"].index(data["default"])
        axis.plot(np.arange(windows), best, "o", color=INK, markersize=3.5)
        static = data["values"].index(data["static_best"])
        axis.plot([windows], [static], "o", color=INK, markersize=3.5)
        axis.set_yticks(range(count))
        axis.set_yticklabels([f"{label} (default)" if i == default else label for i, label in enumerate(data["values"])], fontsize=8.5, color=SECONDARY)
        ticks = sorted({0, windows // 4, windows // 2, (3 * windows) // 4, windows - 1})
        axis.set_xticks([*ticks, windows])
        span = int(bounds[-1] - bounds[0])
        scale, unit = (1000, "k") if span >= 5000 else (1, "")
        axis.set_xticklabels([f"{bounds[i] / scale:g}{unit}" for i in ticks] + ["all"], fontsize=8, color=SECONDARY)
        axis.tick_params(length=0)
        for side in axis.spines.values():
            side.set_visible(False)
        axis.set_title(
            f"{knob}: best static {data['static_best']} {100 * data['static_change']:+.2f}%, best per window {100 * data['switching_change']:+.2f}%",
            loc="left", fontsize=10, color=INK, pad=4,
        )
        top = bottom
    bar = figure.add_axes((0.89, top, 0.015, 1 - 100 / height - top))
    figure.colorbar(image, cax=bar).ax.tick_params(labelsize=8, colors=SECONDARY)
    bar.set_ylabel("change in misses vs s3fifo, %", color=SECONDARY, fontsize=8.5)
    figure.text(0.15, 12 / height, "columns: window start (requests); the last column is the whole range", fontsize=8, color=MUTED)
    return _png(figure)


def render_headroom(profile: dict[str, Any], *, title: str, subtitle: str) -> bytes:
    """Each oracle (rows) x windows (columns) plus the whole range, colored by the
    change in misses vs S3-FIFO; a bar per row gives the whole-range change."""
    rows = profile["rows"]
    names = list(rows)
    row_px = 34
    height = 96 + len(names) * row_px + 60
    figure = _figure(height)
    _title(figure, title, subtitle)
    matrix = np.array([[*rows[name]["relative"], rows[name]["overall"]] for name in names]) * 100
    oracle_rows = [i for i, name in enumerate(names) if name != "OPT" and not rows[name].get("real")]
    finite = np.abs(matrix[oracle_rows].ravel())  # OPT's row would wash out the decisions' colors
    finite = finite[np.isfinite(finite)]
    bound = max(0.5, float(np.quantile(finite, 0.95))) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    bounds = profile["bounds"]
    windows = len(bounds) - 1
    panel = len(names) * row_px / height
    bottom = 60 / height
    axis = figure.add_axes((0.2, bottom, 0.5, panel))
    cmap = DIVERGING.with_extremes(bad=SURFACE)  # real policies have no per-window numbers
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    axis.axvline(windows - 0.5, color=SURFACE, linewidth=4)
    axis.axhline(names.index("OPT") - 0.5, color=SURFACE, linewidth=4)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=9, color=SECONDARY)
    ticks = sorted({0, windows // 4, windows // 2, (3 * windows) // 4, windows - 1})
    axis.set_xticks([*ticks, windows])
    span = int(bounds[-1] - bounds[0])
    scale, unit = (1000, "k") if span >= 5000 else (1, "")
    axis.set_xticklabels([f"{bounds[i] / scale:.3g}{unit}" for i in ticks] + ["all"], fontsize=8, color=SECONDARY)
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    bar = figure.add_axes((0.73, bottom, 0.012, panel))
    figure.colorbar(image, cax=bar).ax.tick_params(labelsize=8, colors=SECONDARY)
    bars = figure.add_axes((0.82, bottom, 0.15, panel))
    overall = [100 * rows[name]["overall"] for name in names]
    positions = np.arange(len(names))
    colors = [EXTRA if rows[name].get("real") else BASELINE if name == "OPT" else BETTER if value < 0 else WORSE for name, value in zip(names, overall, strict=True)]
    bars.barh(positions, overall, color=colors, height=0.6)
    bars.set_ylim(len(names) - 0.5, -0.5)
    low, high = min(0.0, *overall), max(0.0, *overall)
    pad = 0.35 * (high - low or 1.0)  # room for the value labels
    bars.set_xlim(low - (pad if low < 0 else 0), high + (pad if high > 0 else 0))
    bars.set_yticks([])
    bars.axvline(0, color=MUTED, linewidth=0.8)
    for y, value in zip(positions, overall, strict=True):
        bars.annotate(f"{value:+.1f}%", (value, y), xytext=(3 if value >= 0 else -3, 0), textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize=8, color=INK)
    bars.tick_params(labelsize=8, colors=SECONDARY, length=0)
    for side in ("top", "right", "left"):
        bars.spines[side].set_visible(False)
    bars.set_title("whole range", fontsize=9, color=SECONDARY, loc="left")
    figure.text(0.2, 14 / height, "change in misses vs s3fifo, %; columns: window start (requests), the last is the whole range; OPT and real policies (purple) for scale", fontsize=8, color=MUTED)
    return _png(figure)


def render_handover(profile: dict[str, Any], *, name: str, title: str, subtitle: str) -> bytes:
    """Warm takeover vs cold start vs S3-FIFO continuing, and how the inherited cache drains."""
    height = 860
    figure = _figure(height)
    _title(figure, title, subtitle)
    bounds = profile["bounds"]
    x = _centers(bounds)
    start, end = int(bounds[0]), int(bounds[-1])
    axes = figure.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0.42, "left": 0.08, "right": 0.86, "top": 0.90, "bottom": 0.08, "height_ratios": (1.2, 1.0, 0.9)})

    axis = axes[0]
    _style(axis, ylabel="miss ratio")
    axis.set_title(f"Miss ratio per window: s3fifo continuing, {name} taking over (warm), {name} from a cold cache", loc="left", fontsize=10, color=INK, pad=3)
    axis.plot(x, profile["base"], color=MUTED, linewidth=LINE)
    axis.plot(x, profile["cold"], color=EXTRA, linewidth=LINE, linestyle=(0, (5, 2)))
    axis.plot(x, profile["warm"], color=BETTER, linewidth=LINE)
    _label_ends(axis, [(x[-1], profile["base"][-1], "s3fifo", SECONDARY), (x[-1], profile["warm"][-1], "warm", BETTER), (x[-1], profile["cold"][-1], "cold", EXTRA)])

    axis = axes[1]
    _style(axis, ylabel="misses")
    axis.set_title("Cumulative misses beyond s3fifo since the handover (below 0 = fewer)", loc="left", fontsize=10, color=INK, pad=3)
    steps = profile["steps"]
    stride = max(1, steps.size // 2000)
    axis.plot(steps[::stride], profile["cold_excess"][::stride], color=EXTRA, linewidth=LINE, linestyle=(0, (5, 2)))
    axis.plot(steps[::stride], profile["warm_excess"][::stride], color=BETTER, linewidth=LINE)
    axis.axhline(0, color=BASELINE, linewidth=0.8)
    if steps.size:
        _label_ends(axis, [(steps[-1], float(profile["warm_excess"][-1]), f"warm {int(profile['warm_excess'][-1]):+,}", BETTER), (steps[-1], float(profile["cold_excess"][-1]), f"cold {int(profile['cold_excess'][-1]):+,}", EXTRA)])

    axis = axes[2]
    _style(axis, ylabel="per window")
    axis.set_title("Evictions after the handover: inherited objects (inserted before it) vs the new policy's own", loc="left", fontsize=10, color=INK, pad=3)
    width = np.diff(bounds) * 0.86
    axis.bar(x, profile["inherited_evictions"], width=width, color=WORSE, label="inherited")
    axis.bar(x, profile["fresh_evictions"], width=width, bottom=profile["inherited_evictions"], color=BASELINE, label="own")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5, labelcolor=SECONDARY)
    _requests_axis(axis, start, end)
    _handover_marks(axes, [profile["switch"]], start, end)
    return _png(figure)


def render_opt(profile: dict[str, Any], *, name: str, colors: dict[str, str], title: str, subtitle: str, diff: dict[str, Any] | None = None, base_name: str = "", marks: list[int] = ()) -> bytes:
    """A policy against Belady (OPT): which misses were avoidable, and which evictions were good."""
    rows = 4 if diff is not None else 3
    height = 1000 if diff is not None else 820
    figure = _figure(height)
    _title(figure, title, subtitle)
    bounds = profile["bounds"]
    x = _centers(bounds)
    start, end = int(bounds[0]), int(bounds[-1])
    ratios = (1.2, 1.0, 0.9, 0.9)[:rows]
    axes = figure.subplots(rows, 1, sharex=True, gridspec_kw={"hspace": 0.42, "left": 0.08, "right": 0.86, "top": 0.91, "bottom": 0.07, "height_ratios": ratios})
    width = np.diff(bounds) * 0.86

    axis = axes[0]
    _style(axis, ylabel="miss ratio")
    axis.set_title(f"Miss ratio per window: {name} vs OPT (shaded: avoidable misses)", loc="left", fontsize=10, color=INK, pad=3)
    axis.fill_between(x, profile["opt_miss_ratio"], profile["miss_ratio"], color=WORSE, alpha=0.18, linewidth=0)
    axis.plot(x, profile["miss_ratio"], color=INK, linewidth=LINE)
    axis.plot(x, profile["opt_miss_ratio"], color=MUTED, linewidth=LINE, linestyle=(0, (5, 2)))
    _label_ends(axis, [(x[-1], profile["miss_ratio"][-1], name, INK), (x[-1], profile["opt_miss_ratio"][-1], "OPT", SECONDARY)])

    axis = axes[1]
    _style(axis, ylabel="per 1k requests")
    axis.set_title("Avoidable misses (OPT hits) by the queue the key was last evicted from", loc="left", fontsize=10, color=INK, pad=3)
    bottom = np.zeros(len(x))
    for queue, series in profile["avoidable_by_queue"].items():
        values = series * 1000
        axis.bar(x, values, width=width, bottom=bottom, color=colors.get(queue, OTHER), label=f"evicted from {queue}" if queue != "other" else "other")
        bottom += values
    if profile["avoidable_by_queue"]:
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5, labelcolor=SECONDARY)

    axis = axes[2]
    _style(axis, ylabel="good share")
    axis.set_title("Good evictions: share not requested again within the Belady boundary", loc="left", fontsize=10, color=INK, pad=3)
    axis.set_ylim(0, 1.02)
    ends = []
    for queue, series in profile["good_evictions"].items():
        axis.plot(x, series, color=colors.get(queue, OTHER), linewidth=LINE)
        finite = np.flatnonzero(np.isfinite(series))
        if finite.size:
            ends.append((x[finite[-1]], series[finite[-1]], queue, colors.get(queue, SECONDARY)))
    _label_ends(axis, ends)

    if diff is not None:
        axis = axes[3]
        _style(axis, ylabel="per 1k requests")
        axis.set_title(f"Avoidable misses of {base_name}: up = fixed by {name}, down = newly created by {name}", loc="left", fontsize=10, color=INK, pad=3)
        axis.bar(x, diff["fixed"] * 1000, width=width, color=BETTER, label="fixed")
        axis.bar(x, -diff["created"] * 1000, width=width, color=WORSE, label="created")
        axis.axhline(0, color=BASELINE, linewidth=0.8)
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5, labelcolor=SECONDARY)
    _requests_axis(axes[-1], start, end)
    _handover_marks(axes, list(marks), start, end)
    return _png(figure)


def render_workload(profile: dict[str, Any], *, title: str, subtitle: str) -> bytes:
    """Trace characterization: reuse distance and frequency (MQ), one-hit wonders (S3-FIFO)."""
    height = 860
    figure = _figure(height)
    _title(figure, title, subtitle)
    axes = figure.subplots(3, 1, gridspec_kw={"hspace": 0.55, "left": 0.08, "right": 0.86, "top": 0.89, "bottom": 0.07, "height_ratios": (1.15, 0.9, 0.9)})
    capacities = profile["capacities"]

    axis = axes[0]
    reuse = profile["reuse"]
    _style(axis, ylabel="share of requests")
    axis.set_title("Reuse distance: share of requests that re-reference a key within d distinct objects (MQ)", loc="left", fontsize=10, color=INK, pad=3)
    axis.plot(reuse["grid"], reuse["share_within"], color=BLUE_RAMP[5], linewidth=LINE)
    axis.set_xscale("log")
    axis.set_ylim(0, max(0.05, float(np.max(reuse["share_within"])) * 1.12))
    axis.set_xlabel("stack distance d (distinct objects since the previous request to the same key)", color=SECONDARY, fontsize=9)
    for capacity in capacities:
        for value, style in ((capacity, "-"), (0.1 * capacity, ":")):
            axis.axvline(value, color=MUTED, linewidth=0.7, linestyle=style)
        axis.annotate(f"C={capacity:,}", (capacity, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2), textcoords="offset points", fontsize=8, color=SECONDARY, va="top")
    _label_end(axis, reuse["grid"][-1], reuse["share_within"][-1], f"{reuse['share_within'][-1]:.0%} reuse", SECONDARY)
    axis.text(0.01, 0.92, f"{reuse['first_requests_share']:.0%} of requests are first requests", transform=axis.transAxes, fontsize=8.5, color=SECONDARY)

    axis = axes[1]
    frequency = profile["frequency"]
    _style(axis, ylabel="share")
    axis.set_title("Access frequency: objects and requests by an object's request count (MQ)", loc="left", fontsize=10, color=INK, pad=3)
    x = np.arange(len(frequency["labels"]))
    axis.bar(x - 0.2, frequency["object_share"], width=0.38, color=BLUE_RAMP[1], label="share of objects")
    axis.bar(x + 0.2, frequency["request_share"], width=0.38, color=BLUE_RAMP[5], label="share of requests")
    axis.set_xticks(x, [f"{label}" for label in frequency["labels"]])
    axis.set_xlabel("requests per object", color=SECONDARY, fontsize=9)
    axis.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=SECONDARY)

    axis = axes[2]
    wonders = profile["one_hit"]
    _style(axis, ylabel="one-hit share")
    axis.set_title("One-hit wonders: share of a window's objects requested once, windows of N distinct objects (S3-FIFO)", loc="left", fontsize=10, color=INK, pad=3)
    merged: dict[int, tuple[float, list[str]]] = {}
    for label, v in wonders.items():
        if v["one_hit_share"] is not None:
            merged.setdefault(v["objects"], (v["one_hit_share"], []))[1].append(label)
    points = sorted((size, share, "=".join(labels)) for size, (share, labels) in merged.items())
    if points:
        xs, ys, labels = zip(*points, strict=True)
        axis.plot(xs, ys, color=BLUE_RAMP[5], linewidth=LINE, marker="o", markersize=4)
        for x_value, y_value, label in points:
            axis.annotate(f"{label} {y_value:.0%}", (x_value, y_value), xytext=(3, 4), textcoords="offset points", fontsize=7.5, color=SECONDARY)
    axis.set_xscale("log")
    axis.set_ylim(0, 1.05)
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("window size N (distinct objects)", color=SECONDARY, fontsize=9)
    return _png(figure)


def render_mrc(profile: dict[str, Any], *, title: str, subtitle: str) -> bytes:
    """Miss ratio against cache size (log scale) for S3-FIFO, LRU and OPT, with each
    marked policy at C and a guide to the size at which S3-FIFO matches it."""
    height = 560
    figure = _figure(height)
    _title(figure, title, subtitle)
    axis = figure.add_axes((0.08, 0.1, 0.74, 0.76))
    _style(axis, ylabel="miss ratio (requests after the handover)")
    sizes, capacity, ratio = profile["sizes"].astype(float), profile["capacity"], profile["ratio"]
    styles = {"S3-FIFO": (CATEGORICAL[0], "-", "o"), "LRU": (CATEGORICAL[1], "-", "s"), "Belady": (SECONDARY, (0, (4, 3)), None)}
    names = {"S3-FIFO": "S3-FIFO", "LRU": "LRU", "Belady": "OPT (Belady)"}
    ends = []
    for algorithm, (color, style, marker) in styles.items():
        axis.plot(sizes, ratio[algorithm], color=color, linestyle=style, linewidth=LINE, marker=marker, markersize=4.5, label=names[algorithm], zorder=3)
        ends.append((sizes[-1], float(ratio[algorithm][-1]), names[algorithm], SECONDARY))
    axis.set_xscale("log", base=2)
    axis.axvline(capacity, color=MUTED, linewidth=0.8, zorder=1)
    axis.annotate(f"C={capacity:,}", (capacity, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2), textcoords="offset points", fontsize=8, color=SECONDARY, va="top")
    top = max(float(np.max(ratio["LRU"])), float(np.max(ratio["S3-FIFO"])))
    for label, (value, equivalent) in profile["marks"].items():
        top = max(top, value)
        axis.plot([capacity], [value], marker="D", color=INK, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=5)
        text = f"{label}: like S3-FIFO at {equivalent / capacity:.2f} C" if equivalent else f"{label}: outside the sizes shown"
        if equivalent:
            axis.plot([min(capacity, equivalent), max(capacity, equivalent)], [value, value], color=INK, linewidth=0.9, linestyle=":", zorder=4)
            axis.plot([equivalent], [value], marker="o", color=INK, markersize=4, zorder=5)
        axis.annotate(text, (capacity, value), xytext=(-8, 12), textcoords="offset points", ha="right", fontsize=8.5, color=INK,
                      bbox={"boxstyle": "round,pad=0.2", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9}, zorder=6)
    if profile.get("opt_equivalent"):
        axis.text(0.01, 0.02, f"OPT at C does what {profile['opt_equivalent'] / capacity:.2f} C does for S3-FIFO", transform=axis.transAxes, fontsize=8.5, color=SECONDARY)
    axis.set_ylim(0, top * 1.08 if top > 0 else 1)
    axis.set_xlim(sizes[0] / 1.15, sizes[-1] * 1.15)
    axis.set_xticks(sizes)
    axis.set_xticklabels([f"{size / capacity:.3g}C\n{int(size):,}" for size in sizes], fontsize=8, color=SECONDARY)
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_xlabel("cache size (multiple of C, objects)", color=SECONDARY, fontsize=9)
    axis.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=SECONDARY)
    _label_ends(axis, ends)
    return _png(figure)


def render_positions(result: dict[str, Any], *, colors: dict[str, str], title: str, subtitle: str) -> bytes:
    """Hit positions per queue and ghost (S4-FIFO's statistics), small multiples."""
    rows = [(name, data, False) for name, data in result["queues"].items()] + [(name, data, True) for name, data in result["ghosts"].items()]
    header, row_height = 96, 104
    height = header + row_height * len(rows) + 56
    figure = _figure(height)
    _title(figure, title, subtitle)
    labels = result["labels"]
    x = np.arange(len(labels))
    axes = figure.subplots(len(rows), 1, sharex=True, squeeze=False, gridspec_kw={"hspace": 0.55, "left": 0.08, "right": 0.80, "top": 1 - header / height, "bottom": 56 / height})
    for axis, (name, data, ghost) in zip(axes[:, 0], rows, strict=True):
        _style(axis, ylabel="share")
        kind = "ghost hits" if ghost else "hits"
        axis.set_title(f"{name}: {data['hits']:,} {kind}, mean size {data['mean_size']:,.0f}", loc="left", fontsize=9.5, color=INK, pad=2)
        axis.bar(x, data["share"], width=0.8, color=colors.get(name, BLUE_RAMP[4]), hatch="//" if ghost else None, edgecolor=SURFACE if not ghost else colors.get(name, BLUE_RAMP[4]), linewidth=0)
        axis.set_ylim(0, max(0.05, max(data["share"]) * 1.15))
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        median = data["median_position"]
        axis.annotate("no hits" if median is None else f"median {median:.2f}", (1.0, 0.5), xycoords="axes fraction", xytext=(8, 0), textcoords="offset points", fontsize=8.5, color=SECONDARY, va="center")
    axis = axes[-1, 0]
    axis.set_xticks(x, labels)
    axis.set_xlabel("position at hit: entries into the queue since the object entered, in multiples of its mean size (0 = head, ~1 = tail of a FIFO)", color=SECONDARY, fontsize=8.5)
    return _png(figure)
