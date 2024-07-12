import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

SHAPES = ("heart", "star", "square", "infinity")


def resample(z, n):
    """Resample a closed polyline to n points evenly spaced by arc length."""
    closed = np.append(z, z[0])
    s = np.concatenate([[0], np.cumsum(np.abs(np.diff(closed)))])
    target = np.linspace(0, s[-1], n, endpoint=False)
    return np.interp(target, s, closed.real) + 1j * np.interp(target, s, closed.imag)


def shape_points(name, n):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if name == "heart":
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        return x + 1j * y
    if name == "star":
        k = np.arange(10)
        return resample(np.where(k % 2, 0.4, 1.0) * np.exp(1j * (np.pi / 2 + k * np.pi / 5)), n)
    if name == "square":
        return resample(np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]), n)
    if name == "infinity":
        return np.cos(t) + 0.5j * np.sin(2 * t)
    raise ValueError(f"unknown shape {name!r}")


def nearest_neighbor_order(pts):
    n = len(pts)
    visited = np.zeros(n, bool)
    order = np.empty(n, int)
    i = 0
    for k in range(n):
        order[k] = i
        visited[i] = True
        d = np.abs(pts - pts[i])
        d[visited] = np.inf
        i = np.argmin(d)
    return pts[order]


def image_outline(path, max_points=3000, seed=0):
    from PIL import Image

    g = np.asarray(Image.open(path).convert("L"), float) / 255
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    rows, cols = np.nonzero(mag > 0.2 * mag.max()) if mag.max() > 0 else ([], [])
    if len(rows) < 3:
        raise ValueError(f"no edges found in {path}")
    pts = cols - 1j * rows  # image rows grow downward
    if len(pts) > max_points:
        pts = np.random.default_rng(seed).choice(pts, max_points, replace=False)
    return nearest_neighbor_order(pts)


def load_points(path):
    rows = [line.replace(",", " ").split() for line in open(path)]
    xy = np.array([[float(v) for v in r[:2]] for r in rows if r and not r[0].startswith("#")])
    return xy[:, 0] + 1j * xy[:, 1]


def normalize(z):
    z = z - z.mean()
    return z / np.abs(z).max()


def dft(x):
    n = len(x)
    k = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(k, k) / n) @ x


def epicycles(z, n_circles):
    """Return (frequency, coefficient) of the n_circles largest terms, biggest first."""
    n = len(z)
    coef = dft(z) / n
    k = np.arange(n)
    freq = np.where(k <= n // 2, k, k - n)
    idx = np.argsort(-np.abs(coef), kind="stable")[:n_circles]
    return freq[idx], coef[idx]


def chain(freq, coef, t):
    """Joint positions of the epicycle chain at times t in [0, 1]; shape (len(t), n + 1)."""
    terms = coef * np.exp(2j * np.pi * np.outer(t, freq))
    return np.hstack([np.zeros((len(t), 1)), np.cumsum(terms, axis=1)])


def reconstruction_error(z, n_circles):
    freq, coef = epicycles(z, n_circles)
    approx = chain(freq, coef, np.arange(len(z)) / len(z))[:, -1]
    return np.sqrt(np.mean(np.abs(approx - z) ** 2))


def render(z, n_circles, out, frames=200, fps=30):
    freq, coef = epicycles(z, n_circles)
    joints = chain(freq, coef, np.linspace(0, 1, frames))
    tip = joints[:, -1]
    lim = np.abs(joints).max() * 1.05

    fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
    fig.patch.set_facecolor("#111")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_facecolor("#111")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    zc = np.append(z, z[0])
    ax.plot(zc.real, zc.imag, color="#333", lw=1)
    circles = [ax.add_patch(plt.Circle((0, 0), abs(c), fill=False, color="#777", lw=0.6, alpha=0.6))
               for c in coef]
    arm, = ax.plot([], [], color="#bbb", lw=0.8)
    trace, = ax.plot([], [], color="#ffa630", lw=2)

    def update(i):
        for circ, ctr in zip(circles, joints[i, :-1]):
            circ.center = (ctr.real, ctr.imag)
        arm.set_data(joints[i].real, joints[i].imag)
        trace.set_data(tip[: i + 1].real, tip[: i + 1].imag)
        return [*circles, arm, trace]

    FuncAnimation(fig, update, frames=frames, blit=True).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Draw a closed path with Fourier epicycles.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--shape", choices=SHAPES, default="heart")
    src.add_argument("--image", help="trace the edges of this image")
    src.add_argument("--points", help="text file with one 'x y' or 'x,y' pair per line")
    p.add_argument("--circles", type=int, default=50)
    p.add_argument("--samples", type=int, default=512, help="points the path is resampled to")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default="epicycles.gif")
    args = p.parse_args()

    if args.image:
        z = resample(image_outline(args.image), args.samples)
    elif args.points:
        z = resample(load_points(args.points), args.samples)
    else:
        z = shape_points(args.shape, args.samples)
    z = normalize(z)

    print(f"{args.circles} circles, rms error {reconstruction_error(z, args.circles):.4f}")
    render(z, args.circles, args.out, args.frames, args.fps)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
