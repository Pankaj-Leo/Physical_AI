# Incident log

> **MOCK: replies are hand-written fixtures in tests/fixtures/, not Cosmos output.** This file shows the log format and the failure handling only.

Model: `nvidia/Cosmos3-Nano`. Sampling per step is recorded in `incident_log.json` (think steps: temperature 0.6, top_p 0.95; plain steps: temperature 0.7, top_p 0.8; top_k 20). Boxes: model output in 0-1000 coordinates, converted to pixels of the clip size.

| Clip | Incident | Severity | Time (s) | Boxes (px) | Flags |
|---|---|---|---|---|---|
| wh_01 | near_miss | high | 3.20-5.90 | person [515, 229, 753, 620]; forklift [710, 173, 1190, 652] | - |
| wh_02 | near_miss | high | 0.00-12.00 | person and forklift INVALID (outside 0-1000) | timestamp_unparsed_used_full_clip, invalid_box:outside 0-1000 |
| wh_02 | zone_violation | medium | 5.50-9.20 | person inside the restricted zone [765, 148, 911, 461] | - |

## Verdicts per clip

| Clip | Condition | near_miss | zone_violation | fallen_load | Malformed |
|---|---|---|---|---|---|
| wh_01 | lighting=day, camera=ceiling | A | C | C | 0 |
| wh_02 | lighting=night, camera=ceiling | A | A | C | 0 |
| wh_03 | lighting=day, camera=forklift | C | abstain | C | 1 |
| wh_04 | lighting=night, camera=forklift | C | C | B | 0 |

Letters: A = yes (incident), C = no, abstain = (B) or an unparseable reply.

## Evidence (model reasoning, first 400 characters)

- **wh_01 / near_miss**: The worker leaves the walkway at about three seconds and enters the aisle. The forklift is moving and passes him with roughly one meter of clearance. That is within two meters of a moving forklift.
- **wh_02 / near_miss**: A worker walks right next to the forklift near the dock. The forklift lights are on, so it may be moving. The distance is less than two meters.
- **wh_02 / zone_violation**: The floor in front of the robot cell is marked with diagonal hatching. Between about six and nine seconds the worker walks across it.

## Score against labels

These numbers score the hand-written mock replies. They test the scoring code; they say nothing about Cosmos.

Decisions: 12. Precision 0.67, recall 0.67, agreement 0.67, coverage 0.83.

| Slice | n | TP | FP | FN | TN | Abstain | Precision | Recall | Agreement |
|---|---|---|---|---|---|---|---|---|---|
| overall | 12 | 2 | 1 | 1 | 6 | 2 | 0.67 | 0.67 | 0.67 |
| hazard fallen_load | 4 | 0 | 0 | 0 | 3 | 1 | n/a | n/a | 0.75 |
| hazard near_miss | 4 | 1 | 1 | 1 | 1 | 0 | 0.50 | 0.50 | 0.50 |
| hazard zone_violation | 4 | 1 | 0 | 0 | 2 | 1 | 1.00 | 1.00 | 0.75 |
| camera=ceiling | 6 | 2 | 1 | 0 | 3 | 0 | 0.67 | 1.00 | 0.83 |
| camera=forklift | 6 | 0 | 0 | 1 | 3 | 2 | n/a | 0.00 | 0.50 |
| lighting=day | 6 | 1 | 0 | 0 | 4 | 1 | 1.00 | 1.00 | 0.83 |
| lighting=night | 6 | 1 | 1 | 1 | 2 | 1 | 0.50 | 0.50 | 0.50 |

Confusion matrix (rows truth present/absent; columns predicted present/absent/abstain):

`[[2, 1, 1], [1, 6, 1]]`

## Failure modes

- false_positives: wh_02/near_miss
- false_negatives: wh_04/near_miss
- abstentions: wh_03/zone_violation, wh_04/fallen_load
- malformed_replies: wh_03/hazard/zone_violation
- missing_replies: none
- incident_flags: wh_02/near_miss: timestamp_unparsed_used_full_clip, wh_02/near_miss: invalid_box:outside 0-1000

Localization on true positives:

- wh_01/near_miss: temporal IoU 0.793, best box IoU 0.792
- wh_02/zone_violation: temporal IoU 0.811, best box IoU 0.708
