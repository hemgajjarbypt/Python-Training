import pytest
from unittest.mock import patch, MagicMock
from app import get_weather_data, save_weather_data, read_weather_data, main

def test_get_weather_data_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"name": "Ahmedabad", "main": {"temp": 30, "humidity": 50}}
    with patch('requests.get', return_value=mock_response):
        result = get_weather_data("Ahmedabad", "fake_api_key")
        assert result["name"] == "Ahmedabad"
        assert result["main"]["temp"] == 30
        assert result["main"]["humidity"] == 50

def test_get_weather_data_failure():
    mock_response = MagicMock()
    mock_response.status_code = 404
    with patch('requests.get', return_value=mock_response):
        with pytest.raises(Exception) as excinfo:
            get_weather_data("UnknownCity", "fake_api_key")
        assert "Error: 404" in str(excinfo.value)

def test_save_and_read_weather_data(tmp_path):
    data = {"name": "Ahmedabad", "main": {"temp": 30, "humidity": 50}}
    file_path = tmp_path / "weather.json"
    save_weather_data(data, file_path)
    loaded = read_weather_data(file_path)
    assert loaded == data

def test_read_weather_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_weather_data("non_existent_file.json")

def test_save_weather_data_invalid_path():
    data = {"name": "Ahmedabad"}
    with pytest.raises(Exception):
        save_weather_data(data, "/invalid_path/weather.json")

def test_main_success(monkeypatch, tmp_path):
    monkeypatch.setenv('API_KEY', 'fake_api_key')
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"name": "Ahmedabad", "main": {"temp": 30, "humidity": 50}}
    with patch('requests.get', return_value=mock_response):
        with patch('builtins.print') as mock_print:
            with patch('os.getcwd', return_value=str(tmp_path)):
                main()
                mock_print.assert_any_call("Weather data saved to weather.json")
                mock_print.assert_any_call("City:", "Ahmedabad")
                mock_print.assert_any_call("Temperature:", 30)
                mock_print.assert_any_call("Weather:", 50)

def test_main_api_failure(monkeypatch):
    monkeypatch.setenv('API_KEY', 'fake_api_key')
    mock_response = MagicMock()
    mock_response.status_code = 404
    with patch('requests.get', return_value=mock_response):
        with patch('builtins.print') as mock_print:
            main()
            assert any("Error: 404" in str(call.args[0]) for call in mock_print.call_args_list)

def test_main_env_missing(monkeypatch):
    monkeypatch.delenv('API_KEY', raising=False)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"name": "Ahmedabad", "main": {"temp": 30, "humidity": 50}}
    with patch('requests.get', return_value=mock_response):
        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_any_call("Weather data saved to weather.json")
