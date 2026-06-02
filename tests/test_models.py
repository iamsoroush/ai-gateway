def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200

    body = resp.json()
    assert body["object"] == "list"

    by_id = {card["id"]: card for card in body["data"]}
    assert by_id["report-fast"]["provider"] == "gemini"
    assert by_id["report-large"]["provider"] == "openai"
    assert by_id["report-fast"]["object"] == "model"
