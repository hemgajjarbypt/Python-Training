import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os

# Add the current directory to sys.path to import app
sys.path.insert(0, os.path.dirname(__file__))

import app

class TestCleanText:
    def test_clean_text_normal(self):
        text = "This is a normal text."
        result = app.clean_text(text)
        assert result == "This is a normal text."

    def test_clean_text_with_pad(self):
        text = "Text with <pad> token."
        result = app.clean_text(text)
        assert result == "Text with token."

    def test_clean_text_with_eos(self):
        text = "Text with <EOS> token."
        result = app.clean_text(text)
        assert result == "Text with token."

    def test_clean_text_with_both(self):
        text = "Text <pad> with <EOS> tokens."
        result = app.clean_text(text)
        assert result == "Text with tokens."

    def test_clean_text_excessive_whitespace(self):
        text = "Text   with    excessive   whitespace."
        result = app.clean_text(text)
        assert result == "Text with excessive whitespace."

    def test_clean_text_empty(self):
        text = ""
        result = app.clean_text(text)
        assert result == ""

    def test_clean_text_only_tokens(self):
        text = "<pad><EOS>"
        result = app.clean_text(text)
        assert result == ""

class TestInitializeVectorstore:
    @patch('app.os.path.exists')
    @patch('app.FAISS.load_local')
    def test_load_existing_vectorstore(self, mock_load_local, mock_exists):
        mock_exists.return_value = True
        mock_vectorstore = MagicMock()
        mock_load_local.return_value = mock_vectorstore

        app.initialize_vectorstore()

        mock_exists.assert_called_once_with(app.VECTOR_STORE_PATH)
        mock_load_local.assert_called_once_with(app.VECTOR_STORE_PATH, app.embedding_model, allow_dangerous_deserialization=True)
        assert app.vectorstore == mock_vectorstore

    @patch('app.os.path.exists')
    @patch('app.PyPDFLoader')
    @patch('app.RecursiveCharacterTextSplitter')
    @patch('app.FAISS.from_documents')
    @patch('app.FAISS.save_local')
    def test_create_new_vectorstore(self, mock_save_local, mock_from_documents, mock_splitter, mock_loader, mock_exists):
        mock_exists.return_value = False

        # Mock PDF loader
        mock_docs = [MagicMock()]
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_docs
        mock_loader.return_value = mock_loader_instance

        # Mock splitter
        mock_chunk = MagicMock()
        mock_chunk.page_content = "Original text <pad>"
        mock_chunks = [mock_chunk]
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter.return_value = mock_splitter_instance

        # Mock FAISS
        mock_vectorstore = MagicMock()
        mock_from_documents.return_value = mock_vectorstore

        app.initialize_vectorstore()

        mock_exists.assert_called_once_with(app.VECTOR_STORE_PATH)
        mock_loader.assert_called_once_with(app.PDF_PATH)
        mock_loader_instance.load.assert_called_once()
        mock_splitter.assert_called_once_with(chunk_size=500, chunk_overlap=50)
        mock_splitter_instance.split_documents.assert_called_once_with(mock_docs)
        # Check clean_text called on chunks
        assert mock_chunk.page_content == "Original text"
        mock_from_documents.assert_called_once_with(mock_chunks, app.embedding_model)
        mock_vectorstore.save_local.assert_called_once_with(app.VECTOR_STORE_PATH)
        assert app.vectorstore == mock_vectorstore

class TestAskEndpoint:
    def setup_method(self):
        self.client = TestClient(app.app)

    @patch('app.vectorstore')
    @patch('app.qa_model')
    def test_ask_success(self, mock_qa_model, mock_vectorstore):
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Context part 1."
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Context part 2."
        mock_vectorstore.similarity_search.return_value = [mock_doc1, mock_doc2]

        mock_qa_model.return_value = {"answer": "Sample answer"}

        response = self.client.post("/ask", json={"question": "What is this?"})

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is this?"
        assert data["answer"] == "Sample answer"
        assert data["context_snippet"] == "Context part 1. Context part 2...."  # since len <400

    @patch('app.vectorstore')
    @patch('app.qa_model')
    def test_ask_long_context(self, mock_qa_model, mock_vectorstore):
        long_context = "A" * 500
        mock_doc = MagicMock()
        mock_doc.page_content = long_context
        mock_vectorstore.similarity_search.return_value = [mock_doc]

        mock_qa_model.return_value = {"answer": "Answer"}

        response = self.client.post("/ask", json={"question": "Q"})

        assert response.status_code == 200
        data = response.json()
        assert data["context_snippet"] == ("A" * 400) + "..."

    def test_ask_vectorstore_not_initialized(self):
        with patch.object(app, 'vectorstore', None):
            response = self.client.post("/ask", json={"question": "Q"})

            assert response.status_code == 200
            assert "error" in response.json()
            assert response.json()["error"] == "Vector store not initialized."

    @patch('app.vectorstore')
    def test_ask_similarity_search_exception(self, mock_vectorstore):
        mock_vectorstore.similarity_search.side_effect = Exception("Search error")

        response = self.client.post("/ask", json={"question": "Q"})

        assert response.status_code == 200
        assert "error" in response.json()
        assert "Search error" in response.json()["error"]

    @patch('app.vectorstore')
    @patch('app.qa_model')
    def test_ask_qa_model_exception(self, mock_qa_model, mock_vectorstore):
        mock_vectorstore.similarity_search.return_value = [MagicMock(page_content="Context")]
        mock_qa_model.side_effect = Exception("QA error")

        response = self.client.post("/ask", json={"question": "Q"})

        assert response.status_code == 200
        assert "error" in response.json()
        assert "QA error" in response.json()["error"]

    def test_ask_invalid_request_missing_question(self):
        response = self.client.post("/ask", json={})

        assert response.status_code == 422  # Validation error

    def test_ask_invalid_request_wrong_type(self):
        response = self.client.post("/ask", json={"question": 123})

        assert response.status_code == 422
