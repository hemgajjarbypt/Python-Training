import io
import sys
import pytest
from unittest.mock import patch, MagicMock

import Day12.app as app

@pytest.fixture
def mock_dataset():
    return {"text": ["I love this movie!", "This movie is terrible.", "It was okay.", "Best film ever!", "Worst film ever!"]}

@pytest.fixture
def mock_sentiment_results():
    return [
        {"label": "POSITIVE", "score": 0.99},
        {"label": "NEGATIVE", "score": 0.01},
        {"label": "NEUTRAL", "score": 0.50},
        {"label": "POSITIVE", "score": 0.95},
        {"label": "NEGATIVE", "score": 0.05},
    ]

@patch("Day12.app.load_dataset")
@patch("Day12.app.pipeline")
def test_main_prints_sentiments(mock_pipeline, mock_load_dataset, mock_dataset, mock_sentiment_results):
    # Mock the dataset loading
    mock_load_dataset.return_value = mock_dataset

    # Mock the sentiment pipeline to return results in order
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.side_effect = [[result] for result in mock_sentiment_results]
    mock_pipeline.return_value = mock_pipeline_instance

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    app.main()

    sys.stdout = sys.__stdout__

    output = captured_output.getvalue()

    # Check that each review and sentiment label appears in output
    for review, sentiment in zip(mock_dataset["text"], mock_sentiment_results):
        assert review[:512] in output
        assert sentiment["label"] in output
        assert f"(score: {sentiment['score']:.4f})" in output

@patch("Day12.app.load_dataset")
@patch("Day12.app.pipeline")
def test_main_with_empty_dataset(mock_pipeline, mock_load_dataset):
    mock_load_dataset.return_value = {"text": []}
    mock_pipeline.return_value = MagicMock()

    captured_output = io.StringIO()
    sys.stdout = captured_output

    with pytest.raises(ValueError):
        app.main()

    sys.stdout = sys.__stdout__

@patch("Day12.app.load_dataset")
@patch("Day12.app.pipeline")
def test_main_with_short_reviews(mock_pipeline, mock_load_dataset):
    short_texts = {"text": ["Hi", "Ok", "No", "Yes", "Go"]}
    mock_load_dataset.return_value = short_texts

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.side_effect = [[{"label": "POSITIVE", "score": 0.9}]] * 5
    mock_pipeline.return_value = mock_pipeline_instance

    captured_output = io.StringIO()
    sys.stdout = captured_output

    app.main()

    sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    for review in short_texts["text"]:
        assert review in output
        assert "POSITIVE" in output
