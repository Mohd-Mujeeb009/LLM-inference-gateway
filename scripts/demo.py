import json
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer gw_demo_key", "Content-Type": "application/json"}
BODY = {
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "Explain a circuit breaker in two lines"}],
}


def call(path: str, body: dict | None = None) -> tuple[int, dict, dict]:
    data = json.dumps(body).encode() if body else None
    request = urllib.request.Request(BASE + path, data=data, headers=HEADERS)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, dict(response.headers), json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), json.load(error)


for label in ("Normal request", "Identical prompt (cache hit)"):
    status, headers, payload = call("/v1/chat/completions", BODY)
    print(
        f"{label:30} {status} provider={headers.get('X-Provider')} cache={headers.get('X-Cache')}"
    )
status, _, payload = call("/v1/usage")
print(
    f"Usage report                   {status} {payload['requests']} requests, {payload['tokens']} tokens"
)
