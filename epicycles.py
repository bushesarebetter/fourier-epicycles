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






