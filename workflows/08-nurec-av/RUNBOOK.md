# Runbook: the Mcity NuRec digital twin, step by step

What the University of Michigan's Mcity team did to rebuild its 30-acre proving ground as a neural reconstruction,
as told by Mcity's speaker in the SIGGRAPH 2026 talk "AV Simulation with NuRec, Cosmos-Dreams, & 3D Gaussian
Splatting" (transcript `NVIDIA Physical AI | SIGGRAPH 2026/AV Simulation with NuRec, Cosmos-Dreams, & 3D Gaussian
Splatting  Physical AI Day at SIGGRAPH 2026.txt`, 09:10–21:27) and the NuRec slides in the same session.

Rules for this file. Every step names its source. The talk and slides show **no NuRec commands, flags, or container
image names** beyond the box labels "NuRec Reference Container" and "Fixer Container". Where a command would go, this
runbook says **consult NuRec docs**. Steps this repo can do generically are pointed at `prepare_capture.py` and
`train_gsplat.py`; those produce an open Gaussian-splatting layout, not NuRec's NCore input. Auto-captions garble the
names: "Nurek", "NeuraC", "Nuerac", "new wreck" are NuRec; "Encord" is NCore; "PT sync" is PTP sync; "G spot" is
gsplat; "Alpha Sim" is AlpaSim.

## 0. Decide why you need a twin

Mcity's reasons (10:19–12:13): test hazards without real risk; recreate rare edge cases on demand (the speaker's
example: a loose parade balloon blocking traffic); replay a real failure over and over before redeploying; validate
before booking time on the physical track; test cities you cannot drive to.

## 1. Build and calibrate the sensor rig

Source: talk 16:08–16:44.

| Item | As described |
|---|---|
| Vehicle | Ford Mach-E |
| Cameras | six RGB cameras on the roof rack |
| Other sensors | GPS, IMU, LiDAR |
| Time sync | PTP sync (captioned "PT sync") and **camera trigger mode**: "all the cameras fire at the same time", which "is easier for synchronization when we're reconstructing" |
| Calibration | IMU-to-LiDAR, LiDAR-to-camera, camera intrinsics, camera extrinsics |
| Compute | an in-vehicle PC (brand name garbled in the captions) |

Check here: `python prepare_capture.py --capture <folder> --out <prepared>` reports the cross-camera skew per trigger
group and flags groups over a tolerance (default 1 ms) with "check PTP sync and camera trigger mode", plus dropped
frames and pose jumps. It needs frames, poses, and intrinsics exported from your bag first (step 3).

## 2. Drive and record

Mcity records in **ROS bag** (17:05–17:09). Plan the drive for coverage: everything a novel view will need must be
seen by some camera (see `novel_view_envelope.md`). Mcity's advantage on its own track: it can keep dynamic actors off
the road while collecting (15:13–15:21); a city cannot.

Data volume reference: the cost slide lists **800 GB** of S3 storage for this capture (`mcity-cost-comparison.jpg`).

## 3. Convert the recording to NCore

"The most important piece here is converting your recorded data to the Encord [NCore] format. Nvidia has a bunch of
great documentation on how to do that, example conversion scripts. So we just use that to create our own. We record
in ROS bag, so we convert from ROS bag to the Encord data standard" (16:55–17:11). The reference-pipeline slide shows
the same step as "Convert to NCore" for both robots and AVs (`nurec-reference-pipeline.jpg`), and the SDK slide lists
"NCore" among gsplat's core libraries and a "PAI NCore Dataset" (`nurec-sdk-overview.jpg`).

Command: **consult NuRec docs** (the NCore conversion examples). Not shown in the sources.

## 4. Generate auxiliary data

"From there, we generate our auxiliary data. And auxiliary data is like you have your semantic segmentation, ego
masking, so you don't want your vehicle's hood in the reconstructed scene" (17:11–17:24).

Command: **consult NuRec docs**. The open equivalent here: `prepare_capture.py` carries a per-camera ego mask into
`transforms.json` (`mask_path`), and `train_gsplat.py` drops masked pixels from the loss.

## 5. Preview with Instant NuRec

Mcity ran Instant NuRec first, "recommended to us just to ensure that our environment looks good before we spend the
GPU hours to do the full reconstruction". They "got the PLY file, just looked at it in a web browser, verified that
the scene looked good" (17:28–17:49).

What Instant NuRec is, per the slide (`instant-nurec.jpg`): "a feed-forward neural model that takes images as input and
outputs a 3DGS in a single forward pass"; "up to 25% fewer training iterations ... with no fidelity tradeoff vs. full
NRE run"; "single GPU, pip-installable, no NRE dependency". Per the talk (06:40–06:58, caption): a Gaussian-splat PLY
"in less than 2 minutes", usable as the scene or as initialization, on Hugging Face.

Command: **consult NuRec docs** (Instant NuRec model card). Not shown in the sources.

## 6. Run the full reconstruction

Then "we ran it through the reconstruction pipeline, the full one" (17:49–17:52). GPUs: RTX PRO 6000 and H100; an
RTX 4090 also worked when the shared cluster was busy, "took a long time, but ... I was still able to get a
reconstruction" (17:52–18:09). NVIDIA's slide: "Full scene reconstruction in ~53 min on RTX PRO 6000"
(`nurec-libraries.jpg`). The SDK slide names the packaged implementation "NuRec Reference Container", built on gsplat,
Slang, and CUDA (`nurec-sdk-overview.jpg`).

Optional, offline: **Fixer** during reconstruction. The pipeline diagram on the libraries slide labels it "Fixer (Fixer
Container)", with an "Offline" arrow into reconstruction and an "Online" arrow into rendering. Fixer is Cosmos-based,
derived from Difix3D+, and on Hugging Face and GitHub (talk 02:47–03:33, caption).

Command: **consult NuRec docs** (Reference Container, Fixer). Not shown in the sources. Open, non-NuRec alternative on
the same prepared images: `train_gsplat.py` (gsplat; not executed here).

## 7. Export the scene as USDZ

The reconstruction writes **USDZ** ("USDZ (Saved to Host)" on the libraries slide; "USDZ" between Reconstruction and
Rendering on `nurec-reference-pipeline.jpg`). The agent-skills slide describes the same output: "Convert multi-sensor
driving clips into reusable 3DGS scenes in USDZ format" (`nurec-agent-skills.jpg`).

## 8. Validate against the real drive

Mcity compared real footage with the hand-built twin and with the NuRec twin, then subtracted the images: "the dark
pixels are basically similar areas and then the lighter pixels are areas that ... could use some improvement"
(18:12–20:07). The hand-built twin had missing and misplaced assets (a missing flower pot, off-centre light poles);
the NuRec twin had better scale, colour, and placement, with some soft edges they were sharpening by improving
calibration (19:16–19:30, 20:47–20:52).

## 9. Render into a simulator

The reference pipeline serves the scene through a **gRPC API**: the simulation runtime sends a "Request" and receives
"Sensor Data" from the NuRec + gsplat renderer; pre-trained USDZ assets can feed the renderer directly
(`nurec-reference-pipeline.jpg`). Applications on the SDK slide: CARLA and AlpaSim ("gsplat Apps", "fully open source
Apache 2.0") and Isaac Sim plus other Omniverse RTX apps (`nurec-sdk-overview.jpg`). Fixer can also run online, at
render time (talk 03:10–03:21). For edits, the slides show Asset Harvester (insert vehicles harvested from other
drives) and Harmonizer (relight inserted assets); the agent skills add scene editing and sensor rendering.

Command and API schema: **consult NuRec docs**. The sources show the gRPC box, not the service definition.

## 10. Account for cost and time

From the Mcity slide (`mcity-cost-comparison.jpg`): track time $1,300, compute (8 hrs) ~$30, S3 storage (800 GB)
$18.40/mo, salary $960, total ≈ $2,310, against $150,000 and 6 months for the hand-built twin: "~65× cheaper",
"~90× faster". Timeline from the talk (20:26–20:39): "about like a half a day of like data collection and then a day
and a half to ... do the conversion and the reconstruction depending on how much GPUs you have". Reproduce and vary
with `python cost_model.py`.

## 11. What Mcity said comes next

As reported in the talk (20:42–21:23): sharper reconstructions via better calibration, then open-sourcing the NuRec
twin of Mcity; scanning the University of Michigan campus and parts of downtown Ann Arbor; drone data collection for a
low-altitude air corridor from Ann Arbor to Detroit. Separately, the earlier hand-built twin is on GitHub in Omniverse
and CARLA versions (13:29–13:38).
