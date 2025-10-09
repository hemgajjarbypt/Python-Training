import pytest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import extract_text_from_pdf, summarize_text, main, tokenizer, model

class TestExtractTextFromPdf:
    def test_valid_pdf(self):
        # Assuming example.pdf exists and has text
        text = extract_text_from_pdf("example.pdf")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_invalid_pdf_path(self):
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("nonexistent.pdf")

class TestSummarizeText:
    @patch('app.tokenizer')
    @patch('app.model')
    def test_short_text_one_chunk(self, mock_model, mock_tokenizer):
        # Mock tokenizer.encode and decode
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.__call__ = MagicMock(return_value={'input_ids': [1,2,3]})
        mock_tokenizer.decode = MagicMock(return_value="Mocked summary")

        # Mock model.generate
        mock_model.generate = MagicMock(return_value=[[4,5,6]])

        text = "Short text."
        summary = summarize_text(text, max_chunk_size=1000)
        assert summary == "Mocked summary"
        # Check calls: one for chunk, one for final
        assert mock_tokenizer.call_count == 2
        assert mock_model.generate.call_count == 2

    @patch('app.tokenizer')
    @patch('app.model')
    def test_long_text_multiple_chunks(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.__call__ = MagicMock(return_value={'input_ids': [1,2,3]})
        mock_tokenizer.decode = MagicMock(side_effect=["Chunk summary 1", "Chunk summary 2", "Final summary"])

        mock_model.generate = MagicMock(return_value=[[4,5,6]])

        text = "Long text " * 300  # Longer than 1000 chars
        summary = summarize_text(text, max_chunk_size=1000)
        assert summary == "Final summary"
        # Two chunks + final
        assert mock_tokenizer.call_count == 3
        assert mock_model.generate.call_count == 3

    @patch('app.tokenizer')
    @patch('app.model')
    def test_empty_text(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.__call__ = MagicMock(return_value={'input_ids': [1,2,3]})
        mock_tokenizer.decode = MagicMock(return_value="Empty summary")

        mock_model.generate = MagicMock(return_value=[[4,5,6]])

        text = ""
        summary = summarize_text(text)
        assert summary == "Empty summary"
        # Still calls for final summary
        assert mock_tokenizer.call_count == 1
        assert mock_model.generate.call_count == 1

    @patch('app.tokenizer')
    @patch('app.model')
    def test_different_max_chunk_size(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.__call__ = MagicMock(return_value={'input_ids': [1,2,3]})
        mock_tokenizer.decode = MagicMock(side_effect=["Summary 1", "Summary 2", "Final"])

        mock_model.generate = MagicMock(return_value=[[4,5,6]])

        text = "Text " * 200  # About 1000 chars
        summary = summarize_text(text, max_chunk_size=500)
        assert summary == "Final"
        # With 500, more chunks
        assert mock_tokenizer.call_count == 3  # Assuming 2 chunks + final

class TestMain:
    @patch('app.extract_text_from_pdf')
    @patch('app.summarize_text')
    @patch('builtins.print')
    def test_main(self, mock_print, mock_summarize, mock_extract):
        mock_extract.return_value = "Extracted text"
        mock_summarize.return_value = "Summary"

        main()

        mock_extract.assert_called_once_with("example.pdf")
        mock_summarize.assert_called_once_with("Extracted text")
        mock_print.assert_any_call("📄 PDF Summary:\n")
        mock_print.assert_any_call("Summary")
