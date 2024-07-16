import numpy as np
from PIL import Image, ImageDraw

from epicycles import dft, image_outline, reconstruction_error, render, shape_points


def test_dft_matches_numpy():
    rng = np.random.default_rng(0)
    for n in (1, 7, 256, 301):
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        assert np.allclose(dft(x), np.fft.fft(x))


def test_error_drops_as_circles_increase():
    z = shape_points("star", 256)  # corners, so every term matters
    errs = [reconstruction_error(z, n) for n in (1, 4, 16, 64, 256)]
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-9


def test_image_outline_traces_a_disk(tmp_path):
    img = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(img).ellipse((40, 40, 160, 160), fill=255)
    img.save(tmp_path / "disk.png")
    pts = image_outline(tmp_path / "disk.png")
    r = np.abs(pts - pts.mean())
    assert np.all(np.abs(r - 60) < 3)
    # nearest-neighbor ordering should walk the rim, not hop across it
    assert np.median(np.abs(np.diff(pts))) < 2


def test_render_writes_gif(tmp_path):
    out = tmp_path / "a.gif"
    render(shape_points("star", 128), 10, out, frames=5)
    assert Image.open(out).n_frames == 5
