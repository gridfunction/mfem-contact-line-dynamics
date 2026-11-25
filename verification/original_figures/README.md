# Redraw the original numerical figures

From the repository root, use Python with NumPy and Matplotlib installed.

```sh
python3 verification/original_figures/redraw.py --output figures/original
```

The command writes the eight original figure panels as PDFs and PNGs. It uses
the six accepted-step histories in [`../original_histories.csv`](../original_histories.csv)
and the archived geometry in `data/`. It requires neither MFEM nor the VTK
volume snapshots.

- The four 2D JSON files store the ordered phase-field zero contours at
  times 0.02, 0.2, 1.0, and the case's terminal time.
- The two 3D JSON files store the zero contours on the `y=0` symmetry slice
  at the same times and the associated phase-field mass used to draw the
  exact cap. Each file also contains the terminal meridian. The original 3D
  surface comparison revolves this meridian around the vertical axis.
- Time, free energy, energy decay rate, and mass for the history figures
  come directly from the original accepted-step records.

The redraw uses the same plotting functions as the simulation-output
commands in the repository README. The original plots were made with
Matplotlib 3.9.4 and 3.10.8. Output details can vary with Matplotlib and
font versions.

If the original VTK files are available, the extraction command is
`python3 verification/original_figures/export_geometry.py /path/to/Saves`.
Extraction reads the four displayed snapshots from each of the six production
cases. Use the repository's plotting dependencies for extraction.
