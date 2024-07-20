from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict():
    with open("Screenshot 2024-07-20 at 12.22.55.png", "rb") as img:
        response = client.post(
            "/predict",
            files={
                "file": img
            },
            data={
                "k": 5
            }
        )
        assert response.status_code == 200
        json_response = response.json()
        assert "predictions" in json_response
        assert len(json_response["predictions"]) == 5
