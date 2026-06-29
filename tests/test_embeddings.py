"""Route-level tests for the OpenAI-compatible /v1/embeddings endpoint."""

from datetime import datetime, timezone


def test_embeddings_response_shape(fake_client):
    resp = fake_client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "hello"},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["object"] == "list"
    assert body["model"] == "text-embedding-3-small"
    assert body["data"] == [
        {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}
    ]
    assert body["usage"] == {"prompt_tokens": 4, "total_tokens": 4}


def test_embeddings_accepts_batch_input_and_dimensions(fake_client):
    resp = fake_client.post(
        "/v1/embeddings",
        json={
            "model": "text-embedding-3-large",
            "input": ["hello", "world"],
            "encoding_format": "float",
            "dimensions": 256,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "text-embedding-3-large"


def test_embeddings_invalid_body_returns_422(fake_client):
    resp = fake_client.post("/v1/embeddings", json={"input": "missing model"})
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_embeddings_are_recorded(usage_client):
    client, _store = usage_client
    resp = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "hello"},
    )
    assert resp.status_code == 200

    summary = client.get("/v1/usage/summary").json()
    assert summary["requests"] == 1
    assert summary["failed_requests"] == 0
    assert summary["input_tokens"] == 4
    assert summary["output_tokens"] == 0
    assert summary["total_tokens"] == 4
    assert "openai" in summary["cost_by_provider"]
    assert summary["estimated_cost_usd"] == 0.00000008
    assert summary["input_cost_usd"] == 0.00000008
    assert summary["embedding_cost_usd"] == 0.00000008
    assert summary["embedding_cost_by_provider"] == {"openai": 0.00000008}
    assert summary["cost_by_model"] == {"text-embedding-3-small": 0.00000008}
    assert summary["embedding_cost_by_model"] == {"text-embedding-3-small": 0.00000008}

    now = datetime.now(timezone.utc).isoformat()
    requests = client.get("/v1/requests", params={"end": now}).json()["data"]
    assert requests[0]["provider"] == "openai"
    assert requests[0]["provider_model"] == "text-embedding-3-small"
    assert requests[0]["model_alias"] == "text-embedding-3-small"
    assert requests[0]["cost_usd"] == 0.00000008
