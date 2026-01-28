import base64
import json
import time
from io import BytesIO
from typing import List, Optional, Dict, Any
import requests
import numpy as np
from PIL import Image
import itertools
import random
from collections import defaultdict
import threading


class VLACClient:
    """Client for communicating with multiple VLAC services (load-balanced)."""
    
    def __init__(self, service_urls: list[str] = None, timeout: int = 30, max_retries: int = 3, retry_backoff: float = 1.0, shuffle: bool = True, max_concurrency_per_server: int = 1):
        """
        Initialize VLAC client with multiple backend servers.
        
        Args:
            service_urls: List of service URLs (e.g., ["http://localhost:8092", "http://localhost:8093"])
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for a failed request
            retry_backoff: Base backoff (seconds) between retries
            shuffle: If True, shuffle server list at init (avoid always starting at same server)
        """
        if service_urls is None or len(service_urls) == 0:
            raise ValueError("At least one service_url must be provided")
        
        self.service_urls = [url.rstrip("/") for url in service_urls]
        if shuffle:
            random.shuffle(self.service_urls)  # 防止所有worker都从同一台开始
        self.server_cycle = itertools.cycle(self.service_urls)
        
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.session = requests.Session()
        
        # Verify at least one service is alive
        self._health_check()

        self._inflight = defaultdict(int)
        self._lock = threading.Lock()
        self._max_per = max(1, max_concurrency_per_server)
    
    def _pick_server(self) -> str:
        with self._lock:
            # pick server with min inflight (break ties by cyclic order)
            candidates = sorted(self.service_urls, key=lambda u: self._inflight[u])
            for url in candidates:
                if self._inflight[url] < self._max_per:
                    self._inflight[url] += 1
                    return url
            # if all saturated, take the least and still increment
            url = candidates[0]
            self._inflight[url] += 1
            return url
    
    def _release_server(self, url: str):
        with self._lock:
            self._inflight[url] = max(0, self._inflight[url] - 1)
    
    def _health_check(self):
        """Check if at least one VLAC service is alive."""
        for url in self.service_urls:
            try:
                response = self.session.post(f"{url}/healthcheck", timeout=self.timeout)
                response.raise_for_status()
                print(f"✅ VLAC service available at {url}")
                return
            except Exception:
                print(f"⚠️  Service not available at {url}")
        raise RuntimeError("No VLAC services available!")
    
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _post_json(self, endpoint: str, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        timeout = timeout or self.timeout
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            server = self._pick_server()
            url = f"{server}/{endpoint.lstrip('/')}"
            try:
                resp = self.session.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                # if server is overloaded (503 or 429), backoff shorter and try another
                if hasattr(exc, 'response') and exc.response is not None and exc.response.status_code in (429, 503):
                    time.sleep(0.05)
                else:
                    time.sleep(self.retry_backoff * attempt)
            finally:
                self._release_server(server)
        raise last_exc
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image array to base64 string."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
    
    # ------------------------------------------------------------------
    # Public API (和你原来一样，只是内部调用换成 _post_json)
    # ------------------------------------------------------------------
    def check_done(self, task: str, first_frame: np.ndarray, prev_frame: np.ndarray, curr_frame: np.ndarray,
                   reference_frames: Optional[List[np.ndarray]] = None, use_server_reference: bool = True) -> tuple[bool, float]:
        try:
            payload = {
                "task": task,
                "first_frame": self._image_to_base64(first_frame),
                "prev_frame": self._image_to_base64(prev_frame),
                "curr_frame": self._image_to_base64(curr_frame),
                "reference": None,
                "use_server_reference": use_server_reference,
            }
            if reference_frames and len(reference_frames) >= 2:
                payload["reference"] = [self._image_to_base64(ref) for ref in reference_frames]
                payload["use_server_reference"] = False

            result = self._post_json("done", payload, timeout=self.timeout)
            return result.get("done", False), float(result.get("prob", 0.0))
        except Exception as e:
            print(f"VLAC done check failed: {e}")
            return False, 0.0

    def compute_trajectory_values(self, task: str, frames: List[np.ndarray], reference_frames: Optional[List[np.ndarray]] = None,
                                  skip: int = 5, use_server_reference: bool = True, batch_size: int = 8, task_name: Optional[str] = None) -> tuple[List[float], List[float]]:
        try:
            payload = {
                "task": task,
                "frames": [self._image_to_base64(frame) for frame in frames],
                "reference": None,
                "skip": skip,
                "ref_num": 0,
                "batch_size": batch_size,
                "think": False,
                "return_video": False,
                "use_server_reference": use_server_reference,
                "task_name": task_name,
            }
            if reference_frames and len(reference_frames) >= 2:
                payload["reference"] = [self._image_to_base64(ref) for ref in reference_frames]
                payload["ref_num"] = len(reference_frames)
                payload["use_server_reference"] = False
            
            result = self._post_json("trajectory-critic", payload, timeout=max(self.timeout, 60))
            return result.get("value_list", []), result.get("critic_list", [])
        except Exception as e:
            print(f"VLAC trajectory computation failed: {e}")
            return [0.0] * len(frames), [0.0] * (len(frames) - 1)
    
    def pairwise_critic(self, task: str, image_a: np.ndarray, image_b: np.ndarray) -> float:
        try:
            payload = {
                "task": task,
                "image_a": self._image_to_base64(image_a),
                "image_b": self._image_to_base64(image_b),
                "rich": False
            }
            result = self._post_json("pairwise-critic", payload, timeout=self.timeout)
            return float(result.get("critic", 0.0))
        except Exception as e:
            print(f"VLAC pairwise critic failed: {e}")
            return 0.0
