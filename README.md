# kunmeto

RAW-level merging for the Fujifilm GFX100S's multi-shot modes, on Linux and macOS
(Esperanto *kunmeto*, "putting together"):

- **Pixel Shift Multi-Shot:** 16 RAF files → one 400 MP `LinearRaw` DNG. It's an alternative
  to FUJIFILM Pixel Shift Combiner, which runs only on Windows and macOS, and it repairs camera
  shake, wind and hot pixels.
- **Multiple exposure** (average / add / lighten / darken): the camera only saves the blended
  result as a JPEG. kunmeto blends the RAWs instead, into a CFA DNG, with any number of frames
  (the camera stops at 9).

The DNGs open in darktable, RawTherapee and other LibRaw-based tools.

## Install

Needs Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```sh
uvx --from git+https://github.com/SkyLull/kunmeto kunmeto --help    # run without installing
uv tool install git+https://github.com/SkyLull/kunmeto              # or install the `kunmeto` command
```

or with pipx:

```sh
pipx install git+https://github.com/SkyLull/kunmeto
```

## Usage

```sh
kunmeto scan /path/to/100_FUJI                  # list the pixel-shift and multi-exposure sets found
kunmeto shift /path/to/100_FUJI                 # every pixel-shift set -> DSCFxxxx.kunmeto.dng
kunmeto shift --mode color /path/to/100_FUJI    # 4-shot full-colour, 100 MP -> DSCFxxxx.kunmeto-color.dng
kunmeto stack /path/to/100_FUJI                 # every in-camera multi-exposure set
kunmeto stack DSCF0165.JPG                      # one set, picked by its composite JPG
kunmeto stack --mode average A.RAF B.RAF C.RAF  # any RAFs, any count
```

Sets are found from the files' metadata, not their names:
- **Pixel-shift sets:** every frame of a set carries the same image counter.
- **Multiple exposures:** the camera's composite JPG records how many frames went into it and
  which blend mode was used. The RAFs carry no such marker, so when you pass RAF files directly,
  you choose the mode with `--mode` (or at a prompt).

Outputs go next to the first frame (`-o DIR` to change), and existing files are skipped unless
you pass `--force`. A progress bar shows each set's current phase.

| Option | |
|---|---|
| `-o DIR` | output folder |
| `--force` | overwrite existing outputs |
| `--float` | write a 32-bit float DNG instead of 16-bit integer |
| `-j N` | parallel decode processes (default: CPU count) |
| `--dry-run` | show what would be written |
| `--no-repair` | `shift`: the plain merge, without defect repair |
| `--repair-mask` | `shift`: also write `<name>.kunmeto.mask.tif`, one page per frame, white where samples were replaced |
| `--normalize` | `stack`: equalize frame brightness before blending |

## How the pixel-shift merge works

Each of the 16 frames is shifted by a known whole or half pixel, recorded in the RAF's maker
note. kunmeto places every frame's samples directly on a 2× grid, with no interpolation. Every
output pixel then gets one red, two green and one blue sample, all measured rather than
demosaiced. The grid matches Fuji's Combiner output (23296 × 17472).

Then it corrects for what goes wrong between the shots:
- **Brightness drift:** each frame is scaled to the set's median brightness. Otherwise fading
  light during the ~16 exposures shows up as a fine grid.
- **Camera shake and wind:** each frame is compared at low resolution with the median of all
  frames, against a fitted noise model. Where a frame disagrees, for example the first shot
  taken while the camera was still settling, or leaves moving in the wind, its samples are
  dropped there. The gaps are filled from the surviving frames. Where every frame moved, one
  frame fills in.
- **Hot pixels:** a photosite that stands out from its neighbours at the same sensor position
  in many frames is masked. A scene detail moves with the shift; a defect stays put.
- **Highlights:** samples are capped at the level every frame can record, and that level is
  written as the DNG white level. That keeps blown areas uniform instead of turning into a grid.

On an M-series Mac a 16-shot set takes about 45 s.

## Requirements and limits

- **Camera:** only the GFX100S is calibrated. Other models stop with an error; a new model
  needs its grid origin measured once.
- **Memory:** a 16-shot set peaks at about 18 GB of RAM.
- **Multi-exposure mode:** the camera JPG's mode tag is decoded for *lighten* (checked against a
  real file). The *add*, *average* and *darken* values are inferred from the menu order and
  unverified; use `--mode` if a set comes out wrong.
- **Platforms:** Windows isn't tested.

## Tips for darktable

- **Float DNGs look warm:** darktable treats them as non-raw and doesn't apply colour
  calibration automatically. Enable colour calibration, or set white balance to "as shot".
- **`stack --mode add` is always float,** and keeps values above white (the summed brightness).
- **Re-running kunmeto over a DNG darktable already has open** can leave darktable showing the
  old version. Restart darktable, or remove the image and import it again.

## Development

```sh
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

## License

GPL-3.0-or-later. kunmeto isn't affiliated with or endorsed by FUJIFILM Corporation; Fujifilm
and GFX are its trademarks.
