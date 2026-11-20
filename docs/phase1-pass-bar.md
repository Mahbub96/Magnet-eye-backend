# Phase 1 pass bar

Written down **before** the first solver run, as the roadmap requires. Changing a number
here later needs a written reason underneath it, not a silent edit.

## What is tested

Synthetic magnets built with Magpylib, scanned by simulated phone swipes, run through the
same pipeline code the API will use (`app/pipeline`).

| Case | Magnet |
| --- | --- |
| `ring` | Ferrite ring, 40 mm outer / 20 mm inner diameter, 8 mm tall, axial, 0.38 T |
| `bar` | Ferrite bar, 40 × 10 × 10 mm, magnetised along its length, 0.40 T |
| `disc` | NdFeB disc, 20 mm diameter, 5 mm tall, axial, 1.25 T |
| `broken` | The ferrite ring split into three pieces with gaps, one piece flipped, one shifted |
| `two` | Two NdFeB discs 60 mm apart, opposite poles up (reported, not part of the bar) |

## Simulated phone (the "realistic" noise level)

- Magnetometer noise 0.8 µT per axis, range ±4900 µT per axis, 100 Hz.
- Leftover hard-iron error 2 µT, baseline error 1 µT, Earth field present.
- Tracking: 1.5 mm jitter, 2 mm bias per swipe. Orientation: 0.3° jitter, 1° bias per swipe.
- Sensor sits 6 cm from the tracked point inside the phone (offset is applied by the pipeline).
- The phone never goes closer than the height where readings stay under 70 % of the range.

## Scan

Six starting swipes (four parallel, one crossing, one higher), then up to four swipes chosen
by the solver's own next-swipe suggestion. Ten swipes at most.

## Pass bar (realistic noise, after the guided swipes)

1. **Field error ≤ 10 %**: relative RMS error of the magnet's field over a test volume
   inside the scanned footprint, from the scan height up to 1.5× the scan height.
2. **Moment error ≤ 15 %**: total magnetic moment (vector) of the reconstruction compared
   with the truth (`ring`, `bar`, `disc`).
3. **Poles of the broken magnet**: each piece's summed reconstructed moment points the same
   way as the truth (cosine ≥ 0.7).
4. **Honest uncertainty**: at least 80 % of test-point field errors fall inside the reported
   2σ band.
5. **Speed**: one solve, including choosing the regularisation strength, in under 1 s on a
   laptop (the 1-core VPS budget is checked in Phase 4).

Not in the bar: seeing a ring's hole or the exact outline. Detail finer than the distance from
the sensor to the magnet cannot be recovered (see the Challenges guide, section 4.4), so the
report shows it but does not grade it.
