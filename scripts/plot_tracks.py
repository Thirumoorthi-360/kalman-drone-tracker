#!/usr/bin/env python3
"""Plot the CSV output of drone_tracker.

    python scripts/plot_tracks.py output                 # saves output/tracks.png
    python scripts/plot_tracks.py output --animate       # also saves output/tracking.gif

Needs: numpy, matplotlib (and pillow for --animate).
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
          "#0891b2", "#be185d", "#4d7c0f", "#a16207", "#475569"]


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def load(d):
    scene = {r["key"]: (r["xmin"], r["xmax"], r["ymin"], r["ymax"]) for r in read_csv(os.path.join(d, "scene.csv"))}
    by_frame = lambda rows: _group(rows)  # noqa: E731
    return dict(
        scene=scene,
        truth=by_frame(read_csv(os.path.join(d, "truth.csv"))),
        dets=by_frame(read_csv(os.path.join(d, "detections.csv"))),
        tracks=by_frame(read_csv(os.path.join(d, "tracks.csv"))),
        fov={int(r["frame"]): r for r in read_csv(os.path.join(d, "fov.csv"))},
    )


def _group(rows):
    g = defaultdict(list)
    for r in rows:
        g[int(r["frame"])].append(r)
    return g


def draw_map(ax, scene):
    ax.add_patch(Rectangle((0, 136), 400, 24, color="#e5e7eb", zorder=0))
    ax.add_patch(Rectangle((190, 0), 20, 300, color="#e5e7eb", zorder=0))
    tx0, tx1, ty0, ty1 = scene["trees"]
    ax.add_patch(Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, color="#16a34a", alpha=0.35, zorder=5, lw=0))
    ax.text((tx0 + tx1) / 2, ty1 + 4, "tree cover", ha="center", fontsize=9, color="#166534")
    ax0, ax1, ay0, ay1 = scene["area"]
    ax.set_xlim(ax0, ax1)
    ax.set_ylim(ay0, ay1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def color(label):
    return COLORS[(int(label) - 1) % len(COLORS)]


def plot_before_after(data, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax in (a1, a2):
        draw_map(ax, data["scene"])
    pts = np.array([[r["x"], r["y"]] for rows in data["dets"].values() for r in rows])
    a1.scatter(pts[:, 0], pts[:, 1], s=4, color="#374151", alpha=0.5)
    a1.set_title("What the drone sees: raw detections")

    paths = defaultdict(list)
    for k in sorted(data["tracks"]):
        for r in data["tracks"][k]:
            if r["confirmed"]:
                paths[int(r["label"])].append((r["x"], r["y"], r["updated"]))
    for lab, p in sorted(paths.items()):
        p = np.array(p)
        for a, b in zip(p[:-1], p[1:]):
            a2.plot([a[0], b[0]], [a[1], b[1]], color=color(lab), lw=2.2, ls="-" if b[2] else ":")
        a2.text(p[-1, 0] + 3, p[-1, 1] + 3, f"#{lab}", color=color(lab), fontsize=10, weight="bold")
    a2.plot([], [], color="#374151", ls=":", label="coasting (no detection)")
    a2.legend(loc="upper right")
    a2.set_title("What the tracker outputs: labelled tracks")
    plt.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out)


def ellipse(x, y, pxx, pxy, pyy, **kw):
    vals, vecs = np.linalg.eigh(np.array([[pxx, pxy], [pxy, pyy]]))
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 4 * np.sqrt(np.maximum(vals[::-1], 1e-9))  # 2-sigma
    return Ellipse((x, y), w, h, angle=angle, fill=False, **kw)


def animate(data, out):
    frames = sorted(data["fov"])
    trails = defaultdict(list)
    fig, ax = plt.subplots(figsize=(10, 7.5))

    def draw(k):
        ax.clear()
        draw_map(ax, data["scene"])
        f = data["fov"][k]
        ax.add_patch(Rectangle((f["xmin"], f["ymin"]), f["xmax"] - f["xmin"], f["ymax"] - f["ymin"],
                               fill=False, ls="--", ec="#6b7280"))
        for r in data["truth"].get(k, []):
            ax.plot(r["x"], r["y"], marker="s", ms=12, mfc="none", mec="#9ca3af")
        for r in data["dets"].get(k, []):
            ax.plot(r["x"], r["y"], "x" if r["is_clutter"] else "o", color="#374151", ms=5)
        n = 0
        for r in data["tracks"].get(k, []):
            if not r["confirmed"]:
                continue
            n += 1
            lab, c = int(r["label"]), color(r["label"])
            trails[lab].append((r["x"], r["y"]))
            t = np.array(trails[lab][-24:])
            ax.plot(t[:, 0], t[:, 1], color=c, lw=1.5, alpha=0.6)
            ax.plot(r["x"], r["y"], "s", ms=9, color=c, mfc=c if r["updated"] else "white", mew=2, zorder=7)
            ax.add_patch(ellipse(r["x"], r["y"], r["pxx"], r["pxy"], r["pyy"], ec=c, lw=1.5,
                                 ls="-" if r["updated"] else "--"))
            ax.annotate("", xy=(r["x"] + 1.5 * r["vx"], r["y"] + 1.5 * r["vy"]), xytext=(r["x"], r["y"]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
            ax.text(r["x"] + 5, r["y"] + 5, f"#{lab}" + ("" if r["updated"] else " coasting"),
                    color=c, fontsize=10, weight="bold", zorder=8)
        ax.set_title(f"t = {f['t']:4.1f} s   |   detections: {len(data['dets'].get(k, []))}   |   tracks: {n}")

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=150)
    anim.save(out, writer=animation.PillowWriter(fps=7), dpi=70)
    plt.close(fig)
    print("saved", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="?", default="output", help="folder containing drone_tracker CSV output")
    p.add_argument("--animate", action="store_true", help="also save an animated GIF")
    a = p.parse_args()
    data = load(a.dir)
    plot_before_after(data, os.path.join(a.dir, "tracks.png"))
    if a.animate:
        animate(data, os.path.join(a.dir, "tracking.gif"))


if __name__ == "__main__":
    main()
