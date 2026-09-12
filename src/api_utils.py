# api_utils.py
import json
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:2026"
ROBOT_ID = "202619058021"  # ⚠️ 确保这是你们真实的队号

def post(path, payload):
    request = Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as http_response:
            return json.loads(http_response.read().decode("utf-8"))
    except Exception as e:
        print(f"❌ 请求 {path} 异常: {e}")
        return None

def base(request_id):
    return {"arena_id": "default", "robot_id": ROBOT_ID, "request_id": request_id}

def measure(x, y, channel, req_id):
    payload = base(req_id)
    payload["position"] = {"x": x, "y": y}
    payload["channel"] = channel
    return post("/measure", payload)