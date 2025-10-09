import pytest
from unittest.mock import patch, MagicMock
import app

def test_main_success(capsys):
    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_inputs = MagicMock()
    mock_tokenizer.return_value = mock_inputs  # tokenizer(prompt) returns inputs dict-like
    mock_tokenizer.decode.return_value = "Please provide the report by tomorrow."

    # Mock the model
    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_model.generate.return_value = [mock_outputs]  # generate returns list of tensors

    with patch.object(app, 'tokenizer', mock_tokenizer), \
         patch.object(app, 'model', mock_model):

        app.main()

    # Verify tokenizer was called with the correct prompt
    expected_prompt = "Rewrite the following sentence politely:\nGive me the report by tomorrow."
    mock_tokenizer.assert_called_once_with(expected_prompt, return_tensors="pt")

    # Verify model.generate was called with correct parameters
    mock_model.generate.assert_called_once_with(
        **mock_inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    # Verify decode was called
    mock_tokenizer.decode.assert_called_once_with(mock_outputs, skip_special_tokens=True)

    # Verify the output
    captured = capsys.readouterr()
    assert "Polite Sentence: Please provide the report by tomorrow." in captured.out

def test_main_with_different_output(capsys):
    # Test with different mocked decode output
    mock_tokenizer = MagicMock()
    mock_inputs = MagicMock()
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.decode.return_value = "Could you kindly submit the report by tomorrow?"

    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_model.generate.return_value = [mock_outputs]

    with patch.object(app, 'tokenizer', mock_tokenizer), \
         patch.object(app, 'model', mock_model):

        app.main()

    captured = capsys.readouterr()
    assert "Polite Sentence: Could you kindly submit the report by tomorrow?" in captured.out

# For failure tests, since globals are already loaded, we can't easily test loading failures
# But to cover scenarios, perhaps test if generate raises, but since model is mocked, we can set side_effect
def test_main_generate_failure():
    # Test if generate fails
    mock_tokenizer = MagicMock()
    mock_inputs = MagicMock()
    mock_tokenizer.return_value = mock_inputs

    mock_model = MagicMock()
    mock_model.generate.side_effect = Exception("Generation failed")

    with patch.object(app, 'tokenizer', mock_tokenizer), \
         patch.object(app, 'model', mock_model):
        with pytest.raises(Exception, match="Generation failed"):
            app.main()

def test_main_decode_failure():
    # Test if decode fails
    mock_tokenizer = MagicMock()
    mock_inputs = MagicMock()
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.decode.side_effect = Exception("Decode failed")

    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_model.generate.return_value = [mock_outputs]

    with patch.object(app, 'tokenizer', mock_tokenizer), \
         patch.object(app, 'model', mock_model):
        with pytest.raises(Exception, match="Decode failed"):
            app.main()
