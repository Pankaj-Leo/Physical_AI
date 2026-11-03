"""REST client for the cuOpt microservice container (Wizard #1 runs it on port 8000).

Not executed: needs a running cuOpt server (NVIDIA GPU). Standard library only, so it runs in any container.

The sources show the port (8000, "you click test, success") and a health check, not the HTTP routes or the response
layout. verify: ``ENDPOINTS`` and the response fields against your cuOpt server version; the defaults follow the
self-hosted server's request/poll pattern as we know it.

    python cuopt_client.py --scenario screenshot [--url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from typing import Any

DEFAULT_URL = "http://localhost:8000"
ENDPOINTS = {  # verify
    "health": "/cuopt/health",
    "request": "/cuopt/request",
    "solution": "/cuopt/solution/{req_id}",
}


class CuOptClient:
    def __init__(self, url: str = DEFAULT_URL, timeout_s: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read() or b"{}")

    def health(self) -> bool:
        try:
            self._call("GET", ENDPOINTS["health"])
            return True
        except OSError:
            return False

    def solve(self, payload: dict[str, Any], poll_s: float = 0.5) -> dict[str, Any]:
        """Submit, then poll until a solution arrives or the timeout passes."""
        first = self._call("POST", ENDPOINTS["request"], payload)
        if "response" in first:  # some versions answer synchronously
            return first
        req_id = first.get("reqId") or first.get("req_id")
        if not req_id:
            raise RuntimeError(f"unexpected reply: {first}")
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            out = self._call("GET", ENDPOINTS["solution"].format(req_id=req_id))
            if "response" in out:
                return out
            time.sleep(poll_s)
        raise TimeoutError(f"no solution for {req_id} after {self.timeout_s} s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--scenario", choices=["screenshot", "large"], default="screenshot")
    args = ap.parse_args(argv)

    import cuopt_payload
    import greedy_solver
    import warehouse_graph

    sc = warehouse_graph.screenshot_scenario() if args.scenario == "screenshot" else warehouse_graph.large_scenario()
    payload = cuopt_payload.build_payload(sc)
    client = CuOptClient(args.url)
    if not client.health():
        print(f"cuOpt not reachable at {args.url}; is the container up?")
        return 1
    cu = client.solve(payload)["response"]["solver_response"]
    base = greedy_solver.solve(payload, "total")["response"]["solver_response"]
    print(json.dumps({"cuopt_cost": cu.get("solution_cost"), "baseline_total_cost": base["solution_cost"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
