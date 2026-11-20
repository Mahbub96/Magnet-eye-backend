"""Synthetic test magnets with exactly known fields (Magpylib, SI units).

Every magnet lies on a table whose surface is the plane z = 0, as in a real scan.
"""

from __future__ import annotations

from dataclasses import dataclass

import magpylib as magpy
import numpy as np


@dataclass
class TestMagnet:
    name: str
    sources: magpy.Collection
    radius: float  # rough lateral half-size, m (what a user would eyeball)
    top: float  # height of the magnet's top surface above the table, m

    def field_ut(self, points: np.ndarray) -> np.ndarray:
        return np.atleast_2d(magpy.getB(self.sources, np.atleast_2d(points))) * 1e6

    def pieces(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """(centroid m, moment A·m²) of each separate piece."""
        out = []
        for s in self.sources.sources_all:
            out.append((np.asarray(s.centroid), np.asarray(s.magnetization) * s.volume))
        return out

    @property
    def moment(self) -> np.ndarray:
        return np.sum([m for _, m in self.pieces()], axis=0)


def _ring_segment(phi1, phi2, pol_z, shift=(0.0, 0.0, 0.0)):
    return magpy.magnet.CylinderSegment(
        dimension=(0.010, 0.020, 0.008, phi1, phi2),
        polarization=(0, 0, pol_z),
        position=(shift[0], shift[1], 0.004 + shift[2]),
    )


def ring() -> TestMagnet:
    return TestMagnet("ring", magpy.Collection(_ring_segment(0, 360, 0.38)), 0.020, 0.008)


def bar() -> TestMagnet:
    m = magpy.magnet.Cuboid(
        dimension=(0.040, 0.010, 0.010), polarization=(0.40, 0, 0), position=(0, 0, 0.005)
    )
    return TestMagnet("bar", magpy.Collection(m), 0.020, 0.010)


def disc() -> TestMagnet:
    m = magpy.magnet.Cylinder(
        dimension=(0.020, 0.005), polarization=(0, 0, 1.25), position=(0, 0, 0.0025)
    )
    return TestMagnet("disc", magpy.Collection(m), 0.010, 0.005)


def broken() -> TestMagnet:
    """Ferrite ring broken into three pieces: one flipped over, one pushed 3 mm outward."""
    c = magpy.Collection(
        _ring_segment(0, 115, 0.38),
        _ring_segment(125, 235, -0.38),
        _ring_segment(245, 355, 0.38, shift=(0.0015, -0.0026, 0.0)),
    )
    return TestMagnet("broken", c, 0.022, 0.008)


def two() -> TestMagnet:
    a = magpy.magnet.Cylinder(
        dimension=(0.010, 0.003), polarization=(0, 0, 1.25), position=(-0.03, 0, 0.0015)
    )
    b = magpy.magnet.Cylinder(
        dimension=(0.010, 0.003), polarization=(0, 0, -1.25), position=(0.03, 0, 0.0015)
    )
    return TestMagnet("two", magpy.Collection(a, b), 0.035, 0.003)


ALL = {"ring": ring, "bar": bar, "disc": disc, "broken": broken, "two": two}
