# Recording a frame and running it again

VIctory can write down one frame of the D3D12 work its host layer sends
(with the memory that work reads) in a run of a title, and run that frame
again without the title, in seconds, as often as asked: to see what the
frame costs the GPU, and to try changes on it (a shader left out, shaders
translated again, the submissions sent another way). Code:
`crates/victory-gpu/src/replay`, after the Windows fork's tool of the same
name (avatarmanz/VIctory-7900XTX).

## Recording

Off unless `VICTORY_RECORD` is set: then nothing is hooked, nothing is
read but that variable, once, when the host device is made.

| variable | meaning |
|---|---|
| `VICTORY_RECORD=<folder>` | on; the capture goes to `<folder>\capture-<n>` (outside any source tree: a capture holds the title's data). The title's shader containers go to `<folder>\containers`, for `--retranslate` |
| `VICTORY_RECORD_AT=w<n>` | the frame `n` presents after the world's first frame (the first that ran 500 draws on the host), as `VICTORY_CAPTURE_WORLD` counts them |
| `VICTORY_RECORD_AT=f<n>` or `<seconds>` | the frame of present `n`, or the first one that many seconds after the host device was made |
| file `<folder>\go` | when it appears, the next frame but three is captured (looked for every 16 presents, whatever else is set) |
| `VICTORY_RECORD_FRAMES=<n>` | frames to capture (1) |
| `VICTORY_RECORD_ALWAYS=0` | record lists only from eight frames before the frame asked for, not from the start of the run: for a long session with the recorder waiting for its file, where the work of writing every call down all the time is not wanted. Then a list the title recorded before that and submits in the captured frame is missing (the notes say so) |

With a launcher that takes `VAR=value` arguments, on a Mac (the paths as
the title's process sees them):

    'VICTORY_RECORD=Z:\Users\me\captures\cell' VICTORY_RECORD_AT=w70

Pick a frame that is not one of `VICTORY_CAPTURE_WORLD`'s, nor one of the
two before such a frame: those frames run diagnostics of their own, which
a capture of them would hold.

The calls of every list are written down as the host layer makes them,
from the start of the run (a list is only in a capture if it was recorded
since it was last reset, and a title records lists ahead of the frame that
submits them). The frame asked for is captured, or the first one after it
whose predecessor's lists were all recorded (four frames later at the
most, whatever was recorded): everything sent to the queue between two
presents, the present's own copy to the window included. At the frame's first submission the GPU is
drained and what its memory holds is written out; before every submission
of the frame, what the CPU changed in the memory both sides see is. The
title stands still meanwhile. `frame recorder: the frame is captured` in
the log, and a `frame.bin` whose notes do not say "not whole", mean it is
done; `failed.txt` that it failed. One capture per process.

**Read `notes.txt`** (also in every replay's report): everything that could
not be recorded is listed there, with the capture's numbers: how long the
title stood still, how much was written, what memory it took.

### What it costs

Measured under Wine with D3DMetal 4.0 beta 3 on an M5 Pro, a release
build, a made-up frame (`tests/replay_big.rs`: 1,000 textures of 1.4 MB
and 300 buffers of 1 MB in the GPU's memory, 1 GB of buffers the CPU maps
that views name, a view heap of 2.36 million slots, 20 submissions of
40,000 calls). A title's own numbers are in its capture's notes.

- With the recorder on, a frame took 23.5 ms instead of 18.6 ms: every
  call the host layer makes into a list is written down (some tens of
  bytes, kept until the list is reset). For the whole run it also keeps a
  description of 64 bytes of every view slot of every heap (144 MB for a
  heap of 2.36 million slots), hashes the DXIL of every pipeline made and
  writes it to `shaders.bin`, and writes every shader container of the
  title's to `containers`.
- The captured frame took 0.98 s: the title stood still for 0.5 s in its
  first submission (1.6 GB of the GPU's memory read back and written to
  the disk in 0.3 s, and the 1 GB of memory both sides see) and 0.2 s more
  over the 19 others (20.5 GB of memory both sides see looked at for
  changes, 0.1 s for `frame.bin` 20 times).
- The capture holds nothing of the frame's memory twice. Contents of the
  GPU's memory go through at most three read-back buffers of at most
  256 MB each (765 MB here); of memory both sides see, 8 bytes are kept of
  each 4 KiB page (2 MB for each GB).
- The capture was 2.7 GB: as large as the GPU's memory in use and the
  memory both sides see that is not zero. Loading it and running it three
  times took 1.1 s.

By these rates a title with 8 GB of textures and 6 GB of copies of title
memory that views name would stand still for 4 to 5 s, more than half of
it in the first submission, and leave a capture of 14 GB.

### A title that ends in the middle

`frame.bin` is written anew after the frame's first submission and then
every other second while its submissions go by (its notes say "the frame is
not whole" until the frame's present), and a title that
ends the process itself (`exit`, `abort`, `TerminateProcess`) waits for
the submission being written. So a title whose watchdog ends it after the
long stand-still of the first submission leaves a capture of the
submissions it still made, which replays (without the picture to compare
with); one that is ended from outside during the first submission leaves
none.

Where the build has `VICTORY_NO_HANG_WATCHDOG=1` (docs/macos.md), set it for
a recording run: the title's own watchdog ends it after 30 checks a second
apart without a finished frame, and a capture of some 15 GB written to an
external drive can take that long.

## Running it

    victory replay <folder> [--loops 5] [--timing detailed] [experiments]

On a Mac: `tools/macos/replay.sh <folder> ...`, which runs it under Wine in
a prefix of its own. `<folder>` is the one given as `VICTORY_RECORD` (its
last capture) or a capture. Do not run it while a title runs: they share the
GPU, and the numbers are of both. (In a checkout with `../shared/wine-cx`
and `../shared/run-game.sh` beside it, the script uses that Wine, keeps its
prefix in `../state/<checkout>/wineprefix-replay` and takes the lock a
title's run holds, so the two wait for each other.)

Each loop puts back what the one before changed, then runs the frame as the
runtime does: one submission after the other, each waited for, and between
them the CPU writes the buffers and views the title's CPU had written by
then. The picture goes to `replay.png` in the capture and is compared with
the one presented.

### What the numbers mean

D3DMetal has no timestamp queries (it takes them and answers zero), so all
times are the wall clock's:

- **A submission's time** is from its `ExecuteCommandLists` to the moment
  its fence is reached and the waiting thread runs again: what the
  runtime's submitting thread waits. It holds D3DMetal's own work on the
  lists, the GPU's work, and the way back of the completion through Wine
  (the report's notes say what a submission that does nothing takes: that
  much of each is not the GPU's).
- **The frame's time** is the sum of its submissions (not the time between
  them, which is the title's CPU in a run and next to nothing here). With
  `--no-waits` or `--one-submit` it is from the first
  `ExecuteCommandLists` to the moment everything has run.
- **`--timing detailed`** runs each loop a second time *in pieces*: every
  run of commands of one kind with one pipeline bound becomes a submission
  of its own (with what was bound before it bound again), sent alone and
  waited for. Two numbers come of it for each kind of command and each
  pipeline. *Sent alone* is the pieces' own time, less a submission's that
  does nothing: more than the work costs inside the frame, since on a GPU
  that renders in tiles a pass cut in pieces loads and stores its targets
  for each piece, and nothing overlaps. *Of the frame's own time* shares
  out what each submission took whole among its pieces by those times: the
  shares add up to the frame. Use them to rank; measure what a pipeline
  costs with `--skip-shader ... --compare`.
- Use the median of several loops; the first is slower (pipelines are
  compiled on first use). The report lists every loop.

Pipelines are named as the runtime logs them (`[mesh 0123456789abcdef,
pixel ...]`: the stages and the first 8 bytes of the BLAKE3 of the title's
containers; our own passes `victory: <name>`), with the first 8 bytes of
the BLAKE3 of their DXIL.

### Experiments

| switch | does |
|---|---|
| `--compare` | runs the frame unchanged and with the experiments turn by turn, and prints both and the difference |
| `--skip-shader <text>` | leaves out what runs pipelines with the text in their name or DXIL hash (repeatable) |
| `--pace <ms>` | the GPU stands idle that long after each submission, as it does while a title prepares the next |
| `--no-waits` | all submissions sent at once, one wait at the end; copies on the GPU stand for what the CPU wrote between them |
| `--one-submit` | one `ExecuteCommandLists` for the whole frame |
| `--shaders <folder>` | takes shaders from the folder where it has them: `<hash>.dxil` by the hash the runtime names the title's shader by, or `<hash>@<the pipeline's other hashes>.dxil` for a translation made for one pipeline alone; with `--compare --shaders-b <folder>` a second set, turn by turn with the first |
| `--retranslate <folder>` | only writes that folder: the capture's shaders translated again by this build with the `VICTORY_*` switches of this run, as the runtime would give them to each pipeline |
| `--timing detailed` | times commands (above); `--top <n>` pipelines and pieces are listed |
| `--wait-lists` | each list sent alone and waited for (as `VICTORY_GPU_TIMING` does in a run): the heaviest lists, with what they do |
| `--leave-out <n>` | after the detailed report, for its `n` costliest pipelines: the frame with and without what runs each, turn by turn. What leaving one out saves is what it costs in the frame. (A pass that makes arguments for later calls cannot be left out this way: the calls then read stale ones.) |
| `--skip-clamps` | leaves out the host layer's argument clamps (the passes, what binds them, their barriers); the calls they serve read arguments kept from a first run with them |
| `--no-buffer-barriers`, `--no-barriers` | leaves out the barriers of buffers, or all (no valid D3D12; D3DMetal runs it) |
| `--no-pixels` | every scissor rectangle one texel: draws without their pixels |
| `--binds-only` | nothing that works is sent: what the lists bind, and their barriers |
| `--one-group` | every dispatch is of one thread group: what dispatches cost whatever they compute |
| `--also "<switches>"` | after this run, another of the same capture, loaded once, with these switches (repeatable): `--also "--pace 5" --also "--compare --skip-clamps"` |
| `--names <file>` | names for the title's shaders in the reports: lines `<hash> <name>` |
| `--summary` | only a summary of the capture, read without a GPU, on any host (`cargo run -p victory -- replay <folder> --summary`): resources, what each submission does, which pipelines run how often and into what, the host layer's own passes |
| `--no-restore` | leaves what a loop changed, and makes no copies to restore from |
| `--listing 3..7` | only writes the frame's resources and those lists' commands (no GPU needed, any host) |
| `--image <file>` | where the picture goes |

To measure a change of the translator: record once with the build in use
(its containers are kept), then, with the changed build and its switches,

    tools/macos/replay.sh <folder> --retranslate <shaders>
    tools/macos/replay.sh <folder> --compare --shaders <shaders>

### What a replay takes

A capture of a title's frame (20,000 resources, 5.3 GB of textures, 11.5 GB
of buffers the CPU maps, 1,263 pipelines, 4.2 million view slots) loads in
5 to 7 s once D3DMetal has compiled its pipelines (72 s the first time),
and a loop takes about 0.3 s more than the frame itself. Everything the
frame's views name is made again at its full size, as in the title's run:
D3DMetal counted 24 GB of video memory for it, of which the process itself
has 6 GB resident. Between loops the textures and GPU buffers the frame
changes are put back from copies (2 GB here); the buffers the CPU maps are
looked through page by page, and the pages that are not as at the frame's
start are read from the capture again, so that a loop starts the same
whatever the one before did, experiments included.

All of it is memory the machine cannot page out (27 GB wired while that
capture replays, by 64bitMango's later notes). Do not start one next to a
title or a large build on a 48 GB machine.

## Limits

- Not recorded, but named in the notes when the host layer uses them:
  reserved (tiled) resources, bundles, ray tracing, render passes, enhanced
  barriers, predication, stream output, shading
  rate, creation methods other than `CreateCommittedResource`,
  `CreateCommittedResource3`, `CreatePlacedResource` and
  `CreatePlacedResource2`.
- Not kept: contents of multisampled textures, of textures in memory the
  CPU sees, and of texture levels over 256 MB.
- Texture states come from the barriers seen. A resource that overlaps the
  one last used (aliasing) is taken as without contents.
- Addresses in memory are not rewritten: indirect arguments with addresses
  (vertex buffers, root views) are reported, and would be wrong.
  Descriptors and root arguments are rewritten.
- What the CPU changed in memory both sides see is found by looking at it
  before each submission, page by page, of what that submission reads (all
  of a buffer that a view names). Pages the GPU wrote there since are part
  of it: a replay sets them to the recorded run's results before the next
  submission. Where a frame's shaders give out places in a different order
  from run to run, a replay's picture can differ from the recorded one by
  that.
- Buffers the CPU saw (upload memory, the host layer's copies of title
  memory) are all made the way the host layer makes its copies of title
  memory (`Host::shared_buffer`) in a replay.
- D3DMetal keeps the values, not the bits, of textures of the two packed
  float formats (`R11G11B10_FLOAT`: what is not a number is stored in its
  plain form; `R9G9B9E5_SHAREDEXP`: the exponent becomes the smallest that
  holds the values). A capture has what the title's textures hold by
  then, and a replay's hold the same. Every other format tried comes back
  bit for bit (colour, compressed, depth with stencil, levels, layers,
  volumes: `tests/replay.rs`).
- A replay shares D3DMetal's cache of compiled shaders with the title's
  runs (it is kept by the name of the executable): pipelines the title
  compiled load at once, and a replay's new ones are added to it.

Tests: `tools/macos/test-wine.sh -p victory-gpu --test replay` (a made-up
frame through the host layer, recorded and replayed in every way: the same
picture, buffers and textures of every format), `--test replay_big --
--ignored --nocapture` (what a frame of a title's size costs), and the
format's own in the native suite.
