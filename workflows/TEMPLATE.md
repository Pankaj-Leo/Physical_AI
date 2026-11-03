# NN — Workflow title

> One sentence: what this workflow reproduces, from which tutorial or talk, and which part of Physical AI it shows.

## Source evidence

| Evidence | What it shows |
|---|---|
| ![](../../assets/screens/<file>.jpg) | One line. Name the source (talk, tutorial, slide deck). |

Transcripts used (file names, and which steps each covers):
- `NVIDIA Isaac Sim & Isaac Lab – Complete Tutorial Series/Tutorial #N – ….txt`: steps …

## What the workflow does

Explain the mechanism in your own words, as an engineer would to a colleague: the data flow, the components, why
each step exists. A small ASCII or Mermaid diagram if it helps.

## Code

| File | Purpose | Runs on | Status |
|---|---|---|---|
| `script.py` | … | Isaac Sim 5.x, RTX GPU | not executed: needs GPU (static checks pass) |
| `tests/test_x.py` | … | CPU | executed here |

## Run it (GPU box)

Pinned versions (Isaac Sim / Isaac Lab / ROS 2 distro / Python / driver), then exact commands in order.

## Source step → code map

| Source step (transcript or screenshot) | Code location |
|---|---|
| "Add ROS2 Subscribe Twist node, topic cmd_vel" (Tutorial #6 Part 1) | `script.py: build_graph()` |

## Gotchas from the sources

- Each gotcha, with the source that reported it.

## What we verified here

Exactly what ran in the cloud container (CPU only) and what did not. No claims beyond that.
