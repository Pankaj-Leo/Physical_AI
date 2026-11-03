"""Kit UI test: the window opens and the lighting button creates three lights.

Uses ``omni.kit.test`` (an asyncio unittest) and ``omni.kit.ui_test`` to click the button as a user would.
not executed: needs GPU (Kit). The repo's pytest run ignores this file (see ../../../../../../conftest.py).
"""

import omni.kit.test
import omni.usd
from omni.kit import ui_test


class TestAtlasWindow(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        await omni.usd.get_context().new_stage_async()

    async def test_lighting_button_creates_three_lights(self):
        window = ui_test.find("Physical AI Atlas")
        self.assertIsNotNone(window)
        button = ui_test.find("Physical AI Atlas//Frame/**/Button[*].text=='Randomize lighting'")  # verify: query
        await button.click()
        stage = omni.usd.get_context().get_stage()
        for i in range(3):
            self.assertTrue(stage.GetPrimAtPath(f"/World/AtlasLights/Light_{i:02d}").IsValid())
