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
