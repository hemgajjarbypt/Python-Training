import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the current directory to sys.path to import app
sys.path.insert(0, os.path.dirname(__file__))

import app

class TestExtractTextFromPDF:
    @patch('app.PdfReader')
    def test_extract_text_normal(self, mock_pdf_reader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample text from PDF."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        result = app.extract_text_from_pdf("sample.pdf")
        assert result == "Sample text from PDF."
        mock_pdf_reader.assert_called_once_with("sample.pdf")

    @patch('app.PdfReader')
    def test_extract_text_empty_pdf(self, mock_pdf_reader):
        mock_reader = MagicMock()
        mock_reader.pages = []
        mock_pdf_reader.return_value = mock_reader

        result = app.extract_text_from_pdf("empty.pdf")
        assert result == ""

    @patch('app.PdfReader')
    def test_extract_text_no_text(self, mock_pdf_reader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        result = app.extract_text_from_pdf("no_text.pdf")
        assert result == ""

    @patch('app.PdfReader', side_effect=FileNotFoundError)
    def test_extract_text_file_not_found(self, mock_pdf_reader):
        with pytest.raises(FileNotFoundError):
            app.extract_text_from_pdf("nonexistent.pdf")

class TestChunkText:
    def test_chunk_text_normal(self):
        text = "word1 word2 word3 word4 word5"
        chunks = app.chunk_text(text, chunk_size=2)
        assert chunks == ["word1 word2", "word3 word4", "word5"]

    def test_chunk_text_empty(self):
        chunks = app.chunk_text("", chunk_size=400)
        assert chunks == []

    def test_chunk_text_single_chunk(self):
        text = "word1 word2"
        chunks = app.chunk_text(text, chunk_size=400)
        assert chunks == ["word1 word2"]

    def test_chunk_text_exact_chunk_size(self):
        text = "a " * 400
        chunks = app.chunk_text(text, chunk_size=400)
        assert len(chunks) == 1
        assert len(chunks[0].split()) == 400

    def test_chunk_text_chunk_size_one(self):
        text = "a b c"
        chunks = app.chunk_text(text, chunk_size=1)
        assert chunks == ["a", "b", "c"]

class TestCreateFaissIndex:
    @patch('app.SentenceTransformer')
    @patch('app.faiss.IndexFlatL2')
    @patch('app.np.array')
    def test_create_faiss_index(self, mock_np_array, mock_index_flat_l2, mock_sentence_transformer):
        mock_embedder = MagicMock()
        mock_embeddings = MagicMock()
        mock_embedder.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_embedder

        mock_index = MagicMock()
        mock_index_flat_l2.return_value = mock_index

        mock_np_array.return_value = mock_embeddings

        chunks = ["chunk1", "chunk2"]
        index, embedder = app.create_faiss_index(chunks)

        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        mock_embedder.encode.assert_called_once_with(chunks)
        mock_index_flat_l2.assert_called_once()
        mock_index.add.assert_called_once_with(mock_embeddings)
        assert index == mock_index
        assert embedder == mock_embedder

class TestRetrieveRelevantChunks:
    def test_retrieve_relevant_chunks(self):
        mock_model = MagicMock()
        mock_query_emb = MagicMock()
        mock_model.encode.return_value = mock_query_emb

        mock_index = MagicMock()
        mock_distances = MagicMock()
        mock_indices = MagicMock()
        mock_indices.__getitem__.return_value = [0, 1, 2]
        mock_index.search.return_value = (mock_distances, mock_indices)

        chunks = ["chunk0", "chunk1", "chunk2", "chunk3"]
        result = app.retrieve_relevant_chunks("query", mock_model, mock_index, chunks, top_k=3)

        mock_model.encode.assert_called_once_with(["query"])
        mock_index.search.assert_called_once()
        assert result == ["chunk0", "chunk1", "chunk2"]

class TestAnswerQuestion:
    @patch('app.qa_pipeline')
    def test_answer_question(self, mock_qa_pipeline):
        mock_result = {"answer": "Sample answer"}
        mock_qa_pipeline.return_value = mock_result

        result = app.answer_question("question", "context")
        assert result == "Sample answer"
        mock_qa_pipeline.assert_called_once_with(question="question", context="context")

class TestRagQaFromPdf:
    @patch('app.extract_text_from_pdf')
    @patch('app.chunk_text')
    @patch('app.create_faiss_index')
    @patch('app.retrieve_relevant_chunks')
    @patch('app.answer_question')
    def test_rag_qa_from_pdf(self, mock_answer_question, mock_retrieve, mock_create_index, mock_chunk_text, mock_extract_text):
        mock_extract_text.return_value = "extracted text"
        mock_chunk_text.return_value = ["chunk1", "chunk2"]
        mock_index = MagicMock()
        mock_embedder = MagicMock()
        mock_create_index.return_value = (mock_index, mock_embedder)
        mock_retrieve.return_value = ["relevant chunk"]
        mock_answer_question.return_value = "final answer"

        result = app.rag_qa_from_pdf("pdf_path", "question", top_k=3)

        mock_extract_text.assert_called_once_with("pdf_path")
        mock_chunk_text.assert_called_once_with("extracted text")
        mock_create_index.assert_called_once_with(["chunk1", "chunk2"])
        mock_retrieve.assert_called_once_with("question", mock_embedder, mock_index, ["chunk1", "chunk2"], 3)
        mock_answer_question.assert_called_once_with("question", "relevant chunk")
        assert result == "final answer"

class TestMain:
    @patch('app.rag_qa_from_pdf')
    @patch('builtins.print')
    def test_main(self, mock_print, mock_rag_qa):
        mock_rag_qa.return_value = "answer"

        app.main()

        mock_rag_qa.assert_called_once_with("sample.pdf", "What architecture does this paper introduce?")
        mock_print.assert_called_with("\nAnswer:", "answer")
