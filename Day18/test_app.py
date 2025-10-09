import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from app import load_text_files, main

class TestLoadTextFiles:
    def test_load_text_files_with_txt_files(self, tmp_path):
        folder = tmp_path / "data"
        folder.mkdir()
        (folder / "file1.txt").write_text("content1")
        (folder / "file2.txt").write_text("content2")
        (folder / "file3.md").write_text("content3")
        filenames, texts = load_text_files(str(folder))
        assert set(filenames) == {"file1.txt", "file2.txt"}
        assert set(texts) == {"content1", "content2"}

    def test_load_text_files_empty_folder(self, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()
        filenames, texts = load_text_files(str(folder))
        assert filenames == []
        assert texts == []

    def test_load_text_files_no_txt_files(self, tmp_path):
        folder = tmp_path / "no_txt"
        folder.mkdir()
        (folder / "file1.md").write_text("content1")
        (folder / "file2.json").write_text("content2")
        filenames, texts = load_text_files(str(folder))
        assert filenames == []
        assert texts == []

    def test_load_text_files_folder_not_exist(self):
        with pytest.raises(FileNotFoundError):
            load_text_files("non_existent_folder")

class TestMain:
    @patch('app.load_text_files')
    @patch('builtins.print')
    @patch('faiss.IndexFlatL2')
    @patch('app.model.encode')
    def test_main_success(self, mock_encode, mock_index_class, mock_print, mock_load):
        # Mock load_text_files
        mock_load.return_value = (["file1.txt", "file2.txt"], ["text1", "text2"])

        # Mock encode for texts
        texts_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        query_embedding = np.array([[0.5, 0.6]])
        mock_encode.side_effect = [texts_embeddings, query_embedding]

        # Mock FAISS index
        mock_index = MagicMock()
        mock_index_class.return_value = mock_index
        mock_index.ntotal = 2
        distances = np.array([[0.1, 0.2, 0.3]])
        indices = np.array([[0, 1, 0]])
        mock_index.search.return_value = (distances, indices)

        # Call main
        main()

        # Assertions
        mock_load.assert_called_once_with("Data")
        assert mock_encode.call_count == 2
        mock_encode.assert_any_call(["text1", "text2"], convert_to_tensor=False, show_progress_bar=True)
        mock_encode.assert_any_call(["Which files talk about AI?"])
        mock_index_class.assert_called_once_with(2)
        mock_index.add.assert_called_once()
        mock_index.search.assert_called_once()
        mock_print.assert_any_call("Stored 2 documents in FAISS index.")
        mock_print.assert_any_call("\nTop matching files:\n")
        mock_print.assert_any_call("1. file1.txt  (distance=0.1000)")
        mock_print.assert_any_call("2. file2.txt  (distance=0.2000)")
        mock_print.assert_any_call("3. file1.txt  (distance=0.3000)")
