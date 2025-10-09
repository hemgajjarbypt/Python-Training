import pytest
from unittest.mock import patch, MagicMock
import app

def test_main_success(capsys):
    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_inputs = MagicMock()
    mock_tokenizer.return_value = mock_inputs  # tokenizer(prompt) returns inputs dict-like
    mock_tokenizer.decode.return_value = "Summary text.\nAnother sentence.\nFinal sentence."

    # Mock the model
    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_model.generate.return_value = [mock_outputs]  # generate returns list of tensors

    with patch.object(app, 'tokenizer', mock_tokenizer), \
         patch.object(app, 'model', mock_model):

        app.main()

    # Verify tokenizer was called with the correct prompts
    # For summarize_text
    expected_summarize_prompt = "Summarize the following text in 3 concise sentences:\n\n\n    Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data, \n    recognize patterns, and make decisions with minimal human intervention. \n    From healthcare to finance, AI technologies are automating tasks, improving efficiency, \n    and uncovering insights that were previously impossible to detect.\n    "
    # For extract_keywords
    expected_extract_prompt = "Extract 5 important keywords from this summary:\n\nSummary text.\nAnother sentence.\nFinal sentence.\n\nKeywords:"

    # Since generate is called twice, check calls
    assert mock_tokenizer.call_count == 2  # tokenizer called twice: once for summarize, once for extract
    mock_tokenizer.assert_any_call(expected_summarize_prompt, return_tensors="pt", truncation=True)
    mock_tokenizer.assert_any_call(expected_extract_prompt, return_tensors="pt", truncation=True)

    # Verify model.generate was called with correct parameters twice
    assert mock_model.generate.call_count == 2
    mock_model.generate.assert_called_with(
        **mock_inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7
    )

    # Verify decode was called twice
    assert mock_tokenizer.decode.call_count == 2
    mock_tokenizer.decode.assert_called_with(mock_outputs, skip_special_tokens=True)

    # Verify the output
    captured = capsys.readouterr()
    assert "Summary:\n Summary text.\nAnother sentence.\nFinal sentence." in captured.out
    assert "\nKeywords:\n Summary text.\nAnother sentence.\nFinal sentence." in captured.out

def test_main_with_different_output(capsys):
    # Test with different mocked decode output
    mock_tokenizer = MagicMock()
    mock_inputs = MagicMock()
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.decode.side_effect = ["Different summary.", "AI, machine learning, automation, efficiency, insights"]

    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_model.generate.return_value = [mock_outputs]

    with patch.object(app, 'tokenizer', mock_tokenizer), \
         patch.object(app, 'model', mock_model):

        app.main()

    captured = capsys.readouterr()
    assert "Summary:\n Different summary." in captured.out
    assert "\nKeywords:\n AI, machine learning, automation, efficiency, insights" in captured.out

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

# Additional tests for individual functions
@patch('app.run_model')
def test_summarize_text(mock_run_model):
    mock_run_model.return_value = "Summary"
    result = app.summarize_text("Text")
    assert result == "Summary"
    expected_prompt = "Summarize the following text in 3 concise sentences:\n\nText"
    mock_run_model.assert_called_once_with(expected_prompt)

@patch('app.run_model')
def test_extract_keywords(mock_run_model):
    mock_run_model.return_value = "Keywords"
    result = app.extract_keywords("Summary")
    assert result == "Keywords"
    expected_prompt = "Extract 5 important keywords from this summary:\n\nSummary\n\nKeywords:"
    mock_run_model.assert_called_once_with(expected_prompt)

@patch('app.summarize_text')
@patch('app.extract_keywords')
def test_summarize_and_extract_keywords(mock_extract, mock_summarize):
    mock_summarize.return_value = "Summary"
    mock_extract.return_value = "Keywords"
    summary, keywords = app.summarize_and_extract_keywords("Text")
    assert summary == "Summary"
    assert keywords == "Keywords"
    mock_summarize.assert_called_once_with("Text")
    mock_extract.assert_called_once_with("Summary")

# Edge cases
@patch('app.run_model')
def test_summarize_text_empty(mock_run_model):
    mock_run_model.return_value = ""
    result = app.summarize_text("")
    assert result == ""

@patch('app.run_model')
def test_extract_keywords_empty(mock_run_model):
    mock_run_model.return_value = ""
    result = app.extract_keywords("")
    assert result == ""

@patch('app.summarize_text')
@patch('app.extract_keywords')
def test_summarize_and_extract_keywords_empty(mock_extract, mock_summarize):
    mock_summarize.return_value = ""
    mock_extract.return_value = ""
    summary, keywords = app.summarize_and_extract_keywords("")
    assert summary == ""
    assert keywords == ""
