# PR 3: vectorised `WarpMLS` (TIA distort / stretch / perspective)

- **Branch:** `ErshovVE:perf/warp-mls-vectorise`, from `PaddlePaddle:main` (`dab3fe3`); 3 commits
  (`5a53efd` the change, `e5045d9` the strict test, `d2fa563` black): 2 files.
- **Checked on the branch** (Python 3.12, paddlepaddle 3.3.1 CPU, `.venv-test`):
  - `pytest tests/ppocr -m ""`: 59 passed, 23 of them in the new test file;
  - `black 24.10.0 --check` and `flake8 7.1.1` with the hook's `--select=E9,F63,F7,F82,E721`: clean;
  - the timing below (scratch script, not in the PR).
- **Open:** <https://github.com/ErshovVE/PaddleOCR/pull/new/perf/warp-mls-vectorise>, base
  `PaddlePaddle/PaddleOCR:main`. Copy the title and the body from the raw text of this file (not from a preview),
  into the Write tab.

---

## Title

[Perf] Vectorise WarpMLS used by the TIA augmentations (6–9× faster)

## Body

### Motivation

`RecAug` applies the TIA augmentations (`tia_distort`, `tia_stretch`, `tia_perspective`). All three warp the image
with `WarpMLS` (moving least squares), which is written with Python loops:

- `calc_delta` loops over grid nodes and control points with scalar numpy operations;
- `gen_img` loops over grid cells.

When we profiled the data loading of PP-OCRv6 small rec training (`RecAug` + `RecConAug`), TIA took ~73 % of the
time, and one CPU core gave 11 samples/s. On wide text lines the data loader, not the GPU, set the training speed.

### Change

`ppocr/data/imaug/text_image_aug/warp_mls.py`, same class and the same public interface (`WarpMLS(src, src_pts,
dst_pts, dst_w, dst_h, trans_ratio).generate()`):

- the displacements of all grid nodes are solved at once with numpy broadcasting;
- they are interpolated to every pixel separably (bilinear, the same weights as the loop);
- the image is sampled once with `cv2.remap` (`INTER_LINEAR`, `BORDER_REPLICATE`).

The special cases of the loop are kept: a grid node that coincides with a control point snaps to it; with fewer
than two points the image is returned unwarped.

### Speed

One `WarpMLS.generate()` on TIA distort points (`segment` 6, the maximum `RecAug` uses), 3-channel 48 px high
images, CPU, the same points for both versions:

| Image (h × w) | Loop, ms | Vectorised, ms | Speed-up |
|---|---|---|---|
| 48 × 150 | 4.0 | 0.68 | 5.8× |
| 48 × 320 | 7.5 | 0.84 | 8.9× |
| 48 × 600 | 11.9 | 1.75 | 6.8× |
| 48 × 1200 | 22.9 | 2.68 | 8.6× |
| 48 × 2400 | 45.1 | 6.01 | 7.5× |

### Output

- The displacement field equals the loop's algorithm run in float64 to 1e-6 px, at widths up to 4 000 px.
- The current loop accumulates in float32 and drifts with the coordinate size: it stays within a small tolerance of
  the new field up to 1 600 px, and differs by up to ~0.5 px near the right edge of ~3 000 px wide images.
- Images: `cv2.remap` quantises sub-pixel positions to 1/32 px, so about half of the pixel values differ from the loop
  version by one level and none by more than two. In the timing above the maximum difference was 1 level.

### Tests

`tests/ppocr/test_warp_mls.py` keeps the original loop implementation as the reference (in float32, and in float64
as the exact one):

- the displacement field against the float64 loop at sizes from 16 × 30 to 44 × 4000, including corner points
  that coincide with grid nodes;
- `tia_distort`, `tia_stretch` and `tia_perspective` with the old and the new `WarpMLS`, the same random seed, 1
  and 3 channels: images within 2 levels, mean difference below 0.75;
- fewer than two control points: the image is left unwarped.

```
pytest tests/ppocr/test_warp_mls.py      # 23 passed
```
