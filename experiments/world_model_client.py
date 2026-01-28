import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image


class WorldModelClient:
    """HTTP client for the Cosmos world-model service.

    Sends head RGB frame + action chunk; receives predicted future frames.
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout_s: float = 60.0,
        retries: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    @staticmethod
    def _encode_image_np_to_b64(image_np: np.ndarray) -> str:
        img = Image.fromarray(image_np.astype(np.uint8))
        with io.BytesIO() as output:
            img.save(output, format="PNG")
            return base64.b64encode(output.getvalue()).decode("utf-8")

    @staticmethod
    def _decode_b64_to_image_np(image_b64: str) -> np.ndarray:
        raw = base64.b64decode(image_b64)
        with io.BytesIO(raw) as buf:
            img = Image.open(buf).convert("RGB")
            return np.array(img)

    def healthz(self) -> bool:
        url = f"{self.base_url}/v1/healthz"
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, timeout=self.timeout_s)
                return resp.status_code == 200
            except requests.RequestException:
                time.sleep(self.backoff_s * (2 ** attempt))
        return False

    def metadata(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/metadata"
        resp = self.session.get(url, headers=self.headers, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def predict(
        self,
        image_np: np.ndarray,
        actions_np: np.ndarray,
        episode_id: str,
        step_index: int,
        return_all_frames: bool = True,
    ) -> Tuple[List[np.ndarray], Optional[List[float]], float]:
        """Single-sample predict via /v1/predict.

        Returns (frames_np_list, rewards_or_none, latency_ms).
        """
        assert image_np.ndim == 3 and image_np.shape[2] == 3
        assert actions_np.ndim == 2

        payload = {
            "episode_id": episode_id,
            "step_index": int(step_index),
            "image_b64": self._encode_image_np_to_b64(image_np),
            "actions": actions_np.tolist(),
            "return_all_frames": return_all_frames,
        }
        url = f"{self.base_url}/v1/predict"

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = self.session.post(url, json=payload, headers=self.headers, timeout=self.timeout_s)
                if resp.status_code == 429:
                    time.sleep(self.backoff_s * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                frames_b64 = data.get("frames_b64", [])
                frames = [self._decode_b64_to_image_np(b) for b in frames_b64]
                rewards_raw = data.get("rewards", None)
                rewards = [float(x) for x in rewards_raw] if rewards_raw is not None else None
                latency_ms = float(data.get("latency_ms", 0.0))
                return frames, rewards, latency_ms
            except requests.RequestException as e:
                last_exc = e
                time.sleep(self.backoff_s * (2 ** attempt))
        if last_exc:
            raise last_exc
        raise RuntimeError("WorldModelClient.predict failed without exception")

    def batch_predict(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Batch predict via /v1/batch_predict.

        Each item requires: image_np, actions_np, episode_id, step_index.
        Returns list of dicts with frames (np list), rewards, latency_ms.
        """
        req_items: List[Dict[str, Any]] = []
        for it in items:
            image_np: np.ndarray = it["image_np"]
            actions_np: np.ndarray = it["actions_np"]
            req_items.append(
                {
                    "episode_id": it["episode_id"],
                    "step_index": int(it["step_index"]),
                    "image_b64": self._encode_image_np_to_b64(image_np),
                    "actions": actions_np.tolist(),
                }
            )

        payload = {"items": req_items}
        url = f"{self.base_url}/v1/batch_predict"

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = self.session.post(url, json=payload, headers=self.headers, timeout=self.timeout_s)
                if resp.status_code == 429:
                    time.sleep(self.backoff_s * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                out_items = []
                for out in data.get("items", []):
                    frames_b64 = out.get("frames_b64", [])
                    frames = [self._decode_b64_to_image_np(b) for b in frames_b64]
                    rewards_raw = out.get("rewards", None)
                    rewards = [float(x) for x in rewards_raw] if rewards_raw is not None else None
                    out_items.append(
                        {
                            "frames": frames,
                            "rewards": rewards,
                            "latency_ms": float(out.get("latency_ms", 0.0)),
                        }
                    )
                return out_items
            except requests.RequestException as e:
                last_exc = e
                time.sleep(self.backoff_s * (2 ** attempt))
        if last_exc:
            raise last_exc
        raise RuntimeError("WorldModelClient.batch_predict failed without exception")


import itertools
import threading
import requests
import random

class WorldModelRouterClient:
    """
    Multi-server World Model client with:
      - load balancing across base_urls
      - per-server concurrency caps
      - health checks
      - retries/backoff
      - graceful handling of 429/503 (backpressure)
    API mirrors WorldModelClient for predict() and batch_predict().
    """

    def __init__(
        self,
        service_urls: list[str],
        token: str | None = None,
        timeout_s: float = 200.0,
        retries: int = 3,
        backoff_s: float = 0.2,
        shuffle: bool = True,
        max_concurrency_per_server: int = 1,
    ) -> None:
        if not service_urls:
            raise ValueError("service_urls must be non-empty")
        self.service_urls = [u.rstrip("/") for u in service_urls]
        if shuffle:
            random.shuffle(self.service_urls)
        self.timeout_s = timeout_s
        self.retries = max(1, retries)
        self.backoff_s = max(0.0, backoff_s)
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

        # one Session per server (safer for concurrency)
        self.sessions: dict[str, requests.Session] = {u: requests.Session() for u in self.service_urls}

        # concurrency accounting
        self._inflight: dict[str, int] = {u: 0 for u in self.service_urls}
        self._lock = threading.Lock()
        self._max_per = max(1, max_concurrency_per_server)
        self._cycle = itertools.cycle(self.service_urls)

        # basic health check
        self._health_check()

    # ------------- Static utils (reuse from single-server client) -------------
    @staticmethod
    def _encode_image_np_to_b64(image_np: np.ndarray) -> str:
        img = Image.fromarray(image_np.astype(np.uint8))
        with io.BytesIO() as output:
            img.save(output, format="PNG")
            return base64.b64encode(output.getvalue()).decode("utf-8")

    @staticmethod
    def _decode_b64_to_image_np(image_b64: str) -> np.ndarray:
        raw = base64.b64decode(image_b64)
        with io.BytesIO(raw) as buf:
            img = Image.open(buf).convert("RGB")
            return np.array(img)

    # -------------------------------- Internals --------------------------------
    def _health_check(self) -> None:
        for url in self.service_urls:
            s = self.sessions[url]
            try:
                # try versioned first, then root
                r = s.get(f"{url}/v1/healthz", timeout=self.timeout_s)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            try:
                r = s.get(f"{url}/healthz", timeout=self.timeout_s)
                if r.status_code == 200:
                    return
            except Exception:
                pass
        raise RuntimeError("No world-model services are healthy among: " + ", ".join(self.service_urls))

    def _pick_server(self) -> str:
        with self._lock:
            # prefer server with minimal inflight; break ties by cyclic order
            candidates = sorted(self.service_urls, key=lambda u: self._inflight[u])
            for url in candidates:
                if self._inflight[url] < self._max_per:
                    self._inflight[url] += 1
                    return url
            # if all saturated, take the least and still increment
            url = candidates[0]
            self._inflight[url] += 1
            return url

    def _release_server(self, url: str) -> None:
        with self._lock:
            self._inflight[url] = max(0, self._inflight[url] - 1)

    # ------------------------------ Public API ---------------------------------
    def predict(
        self,
        image_np: np.ndarray,
        actions_np: np.ndarray,
        episode_id: str,
        step_index: int,
        return_all_frames: bool = True,
    ) -> tuple[list[np.ndarray], list[float] | None, float]:
        assert image_np.ndim == 3 and image_np.shape[2] == 3
        assert actions_np.ndim == 2

        payload = {
            "episode_id": episode_id,
            "step_index": int(step_index),
            "image_b64": self._encode_image_np_to_b64(image_np),
            "actions": actions_np.tolist(),
            "return_all_frames": return_all_frames,
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            base = self._pick_server()
            try:
                s = self.sessions[base]
                resp = s.post(f"{base}/v1/predict", json=payload, headers=self.headers, timeout=self.timeout_s)
                # Quick retry on backpressure
                if resp.status_code in (429, 503):
                    time.sleep(0.05)
                    continue
                resp.raise_for_status()
                data = resp.json()
                frames_b64 = data.get("frames_b64", [])
                frames = [self._decode_b64_to_image_np(b) for b in frames_b64]
                rewards_raw = data.get("rewards", None)
                rewards = [float(x) for x in rewards_raw] if rewards_raw is not None else None
                latency_ms = float(data.get("latency_ms", 0.0))
                return frames, rewards, latency_ms
            except requests.RequestException as e:
                last_exc = e
                # exponential backoff
                time.sleep(self.backoff_s * attempt)
            finally:
                self._release_server(base)
        if last_exc:
            raise last_exc
        raise RuntimeError("WorldModelRouterClient.predict failed without exception")

    def batch_predict(
        self,
        items: list[dict[str, Any]],
        shard_across_servers: bool = False,
    ) -> list[dict[str, Any]]:
        """
        items[i]: {
          "image_np": np.ndarray HxWxC,
          "actions_np": np.ndarray [chunk, action_dim],
          "episode_id": str,
          "step_index": int
        }

        If shard_across_servers=False (default), route entire batch to one server.
        If True, shard items evenly across servers to increase throughput, and merge results in order.
        """
        if not items:
            return []

        # Single-server route (simplest and safest with server-side micro-batching)
        if not shard_across_servers:
            req_items = []
            for it in items:
                req_items.append(
                    {
                        "episode_id": it["episode_id"],
                        "step_index": int(it["step_index"]),
                        "image_b64": self._encode_image_np_to_b64(it["image_np"]),
                        "actions": it["actions_np"].tolist(),
                    }
                )
            payload = {"items": req_items}

            last_exc: Exception | None = None
            for attempt in range(1, self.retries + 1):
                base = self._pick_server()
                try:
                    s = self.sessions[base]
                    resp = s.post(f"{base}/v1/batch_predict", json=payload, headers=self.headers, timeout=self.timeout_s)
                    if resp.status_code in (429, 503):
                        time.sleep(0.05)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    out_items = []
                    for out in data.get("items", []):
                        frames_b64 = out.get("frames_b64", [])
                        frames = [self._decode_b64_to_image_np(b) for b in frames_b64]
                        rewards_raw = out.get("rewards", None)
                        rewards = [float(x) for x in rewards_raw] if rewards_raw is not None else None
                        out_items.append(
                            {
                                "frames": frames,
                                "rewards": rewards,
                                "latency_ms": float(out.get("latency_ms", 0.0)),
                            }
                        )
                    return out_items
                except requests.RequestException as e:
                    last_exc = e
                    time.sleep(self.backoff_s * attempt)
                finally:
                    self._release_server(base)
            if last_exc:
                raise last_exc
            raise RuntimeError("WorldModelRouterClient.batch_predict failed without exception")

        # Shard route across servers (parallel)
        # Note: order must be preserved; we gather per-shard then place back.
        order = list(range(len(items)))
        n = len(self.service_urls)
        shards: list[list[int]] = [[] for _ in range(n)]
        for idx, i in enumerate(order):
            shards[idx % n].append(i)

        results: list[dict[str, Any] | None] = [None] * len(items)

        def _call_shard(server_url: str, shard_indices: list[int]):
            if not shard_indices:
                return
            req_items = []
            for i in shard_indices:
                it = items[i]
                req_items.append(
                    {
                        "episode_id": it["episode_id"],
                        "step_index": int(it["step_index"]),
                        "image_b64": self._encode_image_np_to_b64(it["image_np"]),
                        "actions": it["actions_np"].tolist(),
                    }
                )
            payload = {"items": req_items}
            last_exc: Exception | None = None
            for attempt in range(1, self.retries + 1):
                try:
                    s = self.sessions[server_url]
                    resp = s.post(f"{server_url}/v1/batch_predict", json=payload, headers=self.headers, timeout=self.timeout_s)
                    if resp.status_code in (429, 503):
                        time.sleep(0.05)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    outs = data.get("items", [])
                    for local_idx, out in enumerate(outs):
                        frames_b64 = out.get("frames_b64", [])
                        frames = [self._decode_b64_to_image_np(b) for b in frames_b64]
                        rewards_raw = out.get("rewards", None)
                        rewards = [float(x) for x in rewards_raw] if rewards_raw is not None else None
                        results[shard_indices[local_idx]] = {
                            "frames": frames,
                            "rewards": rewards,
                            "latency_ms": float(out.get("latency_ms", 0.0)),
                        }
                    return
                except requests.RequestException as e:
                    last_exc = e
                    time.sleep(self.backoff_s * attempt)
            # mark failures
            for i in shard_indices:
                results[i] = {"frames": [], "rewards": None, "latency_ms": 0.0, "error": str(last_exc) if last_exc else "unknown"}

        # Pick servers for shards (use current inflight state to pick least)
        with self._lock:
            servers_sorted = sorted(self.service_urls, key=lambda u: self._inflight[u])
            for u in servers_sorted:
                self._inflight[u] += 1  # reserve one slot
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.service_urls)) as ex:
                futs = []
                for server_url, shard_indices in zip(servers_sorted, shards):
                    futs.append(ex.submit(_call_shard, server_url, shard_indices))
                for f in futs:
                    f.result()
        finally:
            with self._lock:
                for u in servers_sorted:
                    self._inflight[u] = max(0, self._inflight[u] - 1)

        # fill any None with empty defaults
        for i, r in enumerate(results):
            if r is None:
                results[i] = {"frames": [], "rewards": None, "latency_ms": 0.0}
        return results

