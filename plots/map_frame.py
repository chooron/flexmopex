"""Pixel-accurate axes placement for the map figures."""


def map_axes(fig, rect_in, height, width):
    """Axes at an exact position given in inches as ``(x0, y0, w, h)``."""
    x0, y0, w, h = rect_in
    return fig.add_axes([x0 / width, y0 / height, w / width, h / height])
