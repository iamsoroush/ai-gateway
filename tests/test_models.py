from app.config import MODEL_CATALOG, MODEL_REGISTRY


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200

    body = resp.json()
    assert body["object"] == "list"

    by_id = {card["id"]: card for card in body["data"]}
    assert by_id["report-fast"]["provider"] == "gemini"
    assert by_id["report-large"]["provider"] == "openai"
    assert by_id["report-fast"]["object"] == "model"

    assert set(by_id) == set(MODEL_REGISTRY) | set(MODEL_CATALOG)
    assert len(body["data"]) == len(by_id)

    for model_id, provider in MODEL_CATALOG.items():
        assert by_id[model_id] == {
            "id": model_id,
            "object": "model",
            "provider": provider,
            "provider_model": model_id,
        }


def test_list_models_uses_callable_gemini_ids(client):
    by_id = {card["id"]: card for card in client.get("/v1/models").json()["data"]}

    assert "gemini-3-pro-preview" in by_id
    assert "gemini-3-flash-preview" in by_id
    assert "gemini-3.1-pro-preview" in by_id
    assert "gemini-3-pro" not in by_id
    assert "gemini-3-flash" not in by_id
    assert "gemini-3.1-pro" not in by_id
