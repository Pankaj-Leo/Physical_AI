# Physical AI Atlas Tools

One window, three jobs:

- **Load robot**: reference a robot USD (path or URL) under `/World`. Set the default with `ATLAS_ROBOT_USD`, for
  example the SimReady robot asset from workflow folder `01-openusd-simready`.
- **Randomize lighting**: three sphere lights under `/World/AtlasLights` get a new intensity, colour temperature and
  position on every press.
- **Next variant / Random variant**: switch a variant set on the selected prim (or the last loaded robot). Leave the
  set name empty to pick the first of `material`, `look`, `color`, `paint`, … Variant names are case-sensitive.

Open it from **Window > Physical AI Atlas** if you close it.
