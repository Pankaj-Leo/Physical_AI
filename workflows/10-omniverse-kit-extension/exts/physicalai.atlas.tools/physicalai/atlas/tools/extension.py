"""Physical AI Atlas Tools: an ``omni.ext.IExt`` with an ``omni.ui`` window.

Built the way the Omniverse tutorials build extensions (Introduction 03 and 04, Training Series 005):
- ``on_startup`` / ``on_shutdown`` on an ``omni.ext.IExt`` subclass; Kit calls them when the extension is toggled.
- The window is an ``omni.ui.Window`` with ``VStack`` (column) and ``HStack`` (row) layouts, ``StringField`` inputs
  backed by ``SimpleStringModel``, and ``Button(clicked_fn=lambda: ...)`` callbacks with arguments.
- A "Window" menu entry, because "if we close this extension there is no way to open it again" (Introduction 04).
- Stage edits go through Kit commands where one exists (the Commands window shows them; Introduction 03 and 04),
  else through ``pxr`` in ``usd_ops``.

A message-bus listener (``atlas_switch_variant``) lets a browser page streamed from this app press the variant
button remotely; see ../../../../../streaming.md.

Buttons: load a robot USD (for example the SimReady asset from workflow folder 01) as a reference; randomize lighting;
switch a variant set (next or random) on the selected prim or the last loaded robot, for material/look or
configuration variants.

Pinned: Kit SDK 107.x via the Kit App Template (USD Composer template), Python 3.11; also loads in Isaac Sim 5.0.
not executed: needs GPU (Kit with RTX). The logic is in ``helpers`` and ``usd_ops`` and is tested on CPU.
"""

from __future__ import annotations

import random

import carb
import omni.ext
import omni.kit.commands
import omni.usd
from omni import ui

from . import helpers, usd_ops

WINDOW_TITLE = "Physical AI Atlas"
MENU_PARENT = "Window"


class AtlasToolsExtension(omni.ext.IExt):
    """Entry point named in ``config/extension.toml`` through ``[[python.module]]``."""

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._rng = random.Random(helpers.DEFAULT_SEED)
        self._robot_usd = ui.SimpleStringModel(helpers.default_robot_usd())
        self._variant_set = ui.SimpleStringModel("")  # empty: pick a look-like set automatically
        self._status = ui.SimpleStringModel("ready")
        self._last_robot = ""
        self._window: ui.Window | None = None
        self._menu_items = None
        self._bus_sub = None
        self._build_window()
        self._add_menu()
        self._subscribe_browser_events()
        carb.log_info(f"[{ext_id}] started")

    def on_shutdown(self) -> None:
        self._bus_sub = None  # dropping the subscription object unsubscribes
        self._remove_menu()
        if self._window is not None:
            self._window.destroy()
            self._window = None

    # ---------------------------------------------------------------------------------------------------------------
    # UI
    # ---------------------------------------------------------------------------------------------------------------
    def _build_window(self) -> None:
        self._window = ui.Window(WINDOW_TITLE, width=440, height=280)
        with self._window.frame, ui.VStack(spacing=6, height=0):
            ui.Label("Robot USD (path or URL; default from $ATLAS_ROBOT_USD)")
            ui.StringField(model=self._robot_usd)
            with ui.HStack(spacing=4, height=0):
                ui.Button("Load robot", clicked_fn=self._on_load_robot)
                ui.Button("Randomize lighting", clicked_fn=self._on_randomize_lighting)
            with ui.HStack(spacing=4, height=0):
                ui.Label("Variant set", width=80)
                ui.StringField(model=self._variant_set)
            with ui.HStack(spacing=4, height=0):
                ui.Button("Next variant", clicked_fn=lambda: self._on_switch_variant("next"))
                ui.Button("Random variant", clicked_fn=lambda: self._on_switch_variant("random"))
            ui.StringField(model=self._status, read_only=True)

    def _show_window(self, *_args) -> None:
        if self._window is None:
            self._build_window()
        self._window.visible = True

    def _add_menu(self) -> None:
        try:
            from omni.kit.menu.utils import MenuItemDescription, add_menu_items  # verify: Kit 107 menu API
        except ImportError:
            return
        self._menu_items = [MenuItemDescription(name=WINDOW_TITLE, onclick_fn=self._show_window)]
        add_menu_items(self._menu_items, MENU_PARENT)

    def _remove_menu(self) -> None:
        if not self._menu_items:
            return
        from omni.kit.menu.utils import remove_menu_items

        remove_menu_items(self._menu_items, MENU_PARENT)
        self._menu_items = None

    def _subscribe_browser_events(self) -> None:
        """Listen for ``atlas_switch_variant`` on the Kit message bus, the channel a streamed web page's custom
        messages arrive on (Basic Training 08 used the same mechanism with an OmniGraph "On Message Bus Event" node)."""
        import carb.events
        import omni.kit.app

        bus = omni.kit.app.get_app().get_message_bus_event_stream()  # verify: Kit 107 still exposes this stream
        self._bus_sub = bus.create_subscription_to_pop_by_type(
            carb.events.type_from_string(helpers.VARIANT_EVENT), self._on_browser_event
        )

    def _on_browser_event(self, event) -> None:
        try:
            mode, variant_set = helpers.parse_variant_event(dict(event.payload))
        except ValueError as exc:
            self._say(f"ignored {helpers.VARIANT_EVENT}: {exc}")
            return
        if variant_set:
            self._variant_set.set_value(variant_set)
        self._on_switch_variant(mode)

    def _say(self, text: str) -> None:
        self._status.set_value(text)
        carb.log_info(f"[{self._ext_id}] {text}")

    # ---------------------------------------------------------------------------------------------------------------
    # Button handlers
    # ---------------------------------------------------------------------------------------------------------------
    def _stage(self):
        return omni.usd.get_context().get_stage()

    def _on_load_robot(self) -> None:
        stage = self._stage()
        asset = self._robot_usd.get_value_as_string().strip()
        if not asset:
            self._say("set a robot USD path first")
            return
        base = f"/World/{helpers.prim_name_from_asset(asset)}"
        path = helpers.unique_prim_path({str(p.GetPath()) for p in stage.Traverse()}, base)
        try:
            # Undoable, and the same command the Commands window logs when you drag an asset in (Introduction 04).
            omni.kit.commands.execute(  # verify: CreateReferenceCommand kwargs in Kit 107
                "CreateReferenceCommand",
                usd_context=omni.usd.get_context(),
                path_to=path,
                asset_path=asset,
                instanceable=False,
            )
            prim = stage.GetPrimAtPath(path)
        except Exception as exc:  # command missing or failed: fall back to plain pxr
            carb.log_warn(f"CreateReferenceCommand failed ({exc}); using pxr")
            prim = usd_ops.add_robot_reference(stage, asset)
        self._last_robot = str(prim.GetPath())
        sets = usd_ops.list_variant_sets(prim)
        self._say(f"loaded {self._last_robot}; variant sets: {', '.join(sets) or 'none'}")

    def _on_randomize_lighting(self) -> None:
        params = usd_ops.randomize_lighting(self._stage(), self._rng)
        temps = ", ".join(f"{p['temperature']:.0f} K" for p in params)
        self._say(f"{len(params)} lights: {temps}")

    def _on_switch_variant(self, mode: str) -> None:
        stage = self._stage()
        selected = list(omni.usd.get_context().get_selection().get_selected_prim_paths())
        prim = usd_ops.selected_or_default(stage, selected, self._last_robot)
        if prim is None:
            self._say("select a prim or load a robot first")
            return
        result = usd_ops.switch_variant(prim, self._variant_set.get_value_as_string().strip(), mode, self._rng)
        if result is None:
            self._say(f"{prim.GetPath()} has no matching variant set")
        else:
            self._say(f"{prim.GetPath()}: {result[0]} = {result[1]}")
