import os
import json
import pytest
from unittest import mock
from unittest.mock import MagicMock

import app

@pytest.fixture(autouse=True)
def cleanup_output_file():
    # Cleanup output file before and after tests
    if os.path.exists(app.OUTPUT_FILE):
        os.remove(app.OUTPUT_FILE)
    yield
    if os.path.exists(app.OUTPUT_FILE):
        os.remove(app.OUTPUT_FILE)

def test_main_runs_and_creates_output_file():
    # Mock tokenizer and model to speed up test and avoid downloading models
    dummy_inputs = MagicMock()
    dummy_inputs.to.return_value = dummy_inputs

    dummy_tokenizer = MagicMock()
    dummy_tokenizer.return_value = dummy_inputs

    dummy_model = MagicMock()
    dummy_model.to.return_value = dummy_model
    dummy_model.eval.return_value = None
    dummy_model.return_value = "dummy_output"

    # Mock torch.no_grad as a context manager
    no_grad_mock = MagicMock()
    no_grad_mock.__enter__ = MagicMock(return_value=None)
    no_grad_mock.__exit__ = MagicMock(return_value=None)

    with mock.patch('app.AutoTokenizer.from_pretrained', return_value=dummy_tokenizer), \
         mock.patch('app.AutoModel.from_pretrained', return_value=dummy_model), \
         mock.patch('app.torch.no_grad', return_value=no_grad_mock), \
         mock.patch('app.torch.cuda.is_available', return_value=False), \
         mock.patch('app.torch.device', return_value='cpu'):

        app.main()

    # Check output file created
    assert os.path.exists(app.OUTPUT_FILE)

    # Check contents of output file
    with open(app.OUTPUT_FILE, "r") as f:
        data = json.load(f)

    assert "BERT" in data
    assert "DistilBERT" in data

    for model_name in ["BERT", "DistilBERT"]:
        model_data = data[model_name]
        assert model_data["total_sentences"] == app.NUM_SENTENCES
        assert model_data["total_time_sec"] >= 0
        assert model_data["avg_inference_time_sec"] >= 0
        assert "device" in model_data

def test_main_prints_benchmark_message(capsys):
    dummy_inputs = MagicMock()
    dummy_inputs.to.return_value = dummy_inputs

    dummy_tokenizer = MagicMock()
    dummy_tokenizer.return_value = dummy_inputs

    dummy_model = MagicMock()
    dummy_model.to.return_value = dummy_model
    dummy_model.eval.return_value = None
    dummy_model.return_value = "dummy_output"

    # Mock torch.no_grad as a context manager
    no_grad_mock = MagicMock()
    no_grad_mock.__enter__ = MagicMock(return_value=None)
    no_grad_mock.__exit__ = MagicMock(return_value=None)

    with mock.patch('app.AutoTokenizer.from_pretrained', return_value=dummy_tokenizer), \
         mock.patch('app.AutoModel.from_pretrained', return_value=dummy_model), \
         mock.patch('app.torch.no_grad', return_value=no_grad_mock), \
         mock.patch('app.torch.cuda.is_available', return_value=False), \
         mock.patch('app.torch.device', return_value='cpu'):

        app.main()

    captured = capsys.readouterr()
    assert "✅ Benchmark completed! Results saved to" in captured.out

def test_main_with_cuda_device():
    dummy_inputs = MagicMock()
    dummy_inputs.to.return_value = dummy_inputs

    dummy_tokenizer = MagicMock()
    dummy_tokenizer.return_value = dummy_inputs

    dummy_model = MagicMock()
    dummy_model.to.return_value = dummy_model
    dummy_model.eval.return_value = None
    dummy_model.return_value = "dummy_output"

    no_grad_mock = MagicMock()
    no_grad_mock.__enter__ = MagicMock(return_value=None)
    no_grad_mock.__exit__ = MagicMock(return_value=None)

    with mock.patch('app.AutoTokenizer.from_pretrained', return_value=dummy_tokenizer), \
         mock.patch('app.AutoModel.from_pretrained', return_value=dummy_model), \
         mock.patch('app.torch.no_grad', return_value=no_grad_mock), \
         mock.patch('app.torch.cuda.is_available', return_value=True), \
         mock.patch('app.torch.device', return_value='cuda'):

        app.main()

    with open(app.OUTPUT_FILE, "r") as f:
        data = json.load(f)

    for model_name in ["BERT", "DistilBERT"]:
        assert data[model_name]["device"] == "cuda"
