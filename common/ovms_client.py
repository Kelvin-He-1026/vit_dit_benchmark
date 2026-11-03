"""A minimal client for OpenVINO Model Server's KServe v2 REST API.

Sends a float32 pixel batch as raw bytes (the API's binary-data extension -
JSON would be ~10x larger and slower to encode) and returns the model's one
output as a numpy array. Used by `server_vit_benchmark.py --backend http`
and handy on its own:

    from common.ovms_client import OVMSClient
    client = OVMSClient("http://127.0.0.1:9100", "google-vit-base-patch16-224")
    logits = client.infer(pixel_values_numpy)          # (batch, 1000)
"""

import json

import numpy as np
import requests


class OVMSClient:
    def __init__(self, endpoint, model, timeout=60):
        self.url = f"{endpoint.rstrip('/')}/v2/models/{model}"
        self.session = requests.Session()           # keeps the connection open
        self.timeout = timeout
        meta = self.session.get(self.url, timeout=timeout)
        meta.raise_for_status()
        meta = meta.json()
        self.input_name = meta["inputs"][0]["name"]
        self.output_name = meta["outputs"][0]["name"]

    def infer(self, pixels):
        x = np.ascontiguousarray(pixels, dtype=np.float32)
        header = json.dumps({
            "inputs": [{"name": self.input_name, "shape": list(x.shape), "datatype": "FP32",
                        "parameters": {"binary_data_size": x.nbytes}}],
            "outputs": [{"name": self.output_name, "parameters": {"binary_data": True}}],
        }).encode()
        r = self.session.post(
            f"{self.url}/infer", data=header + x.tobytes(), timeout=self.timeout,
            headers={"Inference-Header-Content-Length": str(len(header)),
                     "Content-Type": "application/octet-stream"})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        n = int(r.headers.get("Inference-Header-Content-Length", len(r.content)))
        meta = json.loads(r.content[:n])
        out = meta["outputs"][0]
        if n < len(r.content):                        # binary output follows the JSON
            data = np.frombuffer(r.content[n:], dtype=np.float32)
        else:
            data = np.asarray(out["data"], dtype=np.float32)
        return data.reshape(out["shape"])
