import pytest
from unittest.mock import patch
from src.crop_management import get_weather_data
from src.model_inference import ModelInference

@patch('src.crop_management.requests.get')
def test_get_weather_data(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"weather": "sunny"}
    result = get_weather_data("Delhi")
    assert result == {"weather": "sunny"}

def test_model_inference():
    mi = ModelInference()
    response = mi.infer("Test query")
    assert isinstance(response, str)
    assert len(response) > 0
