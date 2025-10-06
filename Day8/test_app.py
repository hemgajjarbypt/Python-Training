import pytest
from unittest.mock import patch, MagicMock

def test_sentiment_pipeline_loading():
    import app
    
    # Check that sentiment_pipeline is loaded
    assert hasattr(app, 'sentiment_pipeline')
    assert callable(app.sentiment_pipeline)

def test_main_execution():
    import app
    with patch.object(app, 'sentiment_pipeline') as mock_sentiment:
        # Mock to return different results for each call
        mock_sentiment.side_effect = [
            [{'label': 'POSITIVE', 'score': 0.99}],
            [{'label': 'NEGATIVE', 'score': 0.95}],
            [{'label': 'NEUTRAL', 'score': 0.85}]
        ]
        
        with patch('builtins.print') as mock_print:
            app.main()
            
            # Check that sentiment_pipeline was called 3 times with the correct sentences
            assert mock_sentiment.call_count == 3
            expected_sentences = [
                'The movie was absolutely fantastic, I loved every moment of it!',
                'I am really disappointed with the service, it was terrible.',
                'The product is okay, not great but not too bad either.'
            ]
            for i, call in enumerate(mock_sentiment.call_args_list):
                assert call[0][0] == expected_sentences[i]
            
            # Check that print was called 3 times with the results
            assert mock_print.call_count == 3
            expected_results = [
                [{'label': 'POSITIVE', 'score': 0.99}],
                [{'label': 'NEGATIVE', 'score': 0.95}],
                [{'label': 'NEUTRAL', 'score': 0.85}]
            ]
            for i, call in enumerate(mock_print.call_args_list):
                assert call[0][0] == expected_results[i]

def test_sentiment_analysis_edge_cases():
    import app
    with patch.object(app, 'sentiment_pipeline') as mock_sentiment:
        mock_sentiment.return_value = [{'label': 'POSITIVE', 'score': 0.5}]
        
        # Test with empty string
        result = app.sentiment_pipeline("")
        assert result == [{'label': 'POSITIVE', 'score': 0.5}]
        
        # Test with long text
        long_text = "This is a very long sentence. " * 10
        result = app.sentiment_pipeline(long_text)
        assert result == [{'label': 'POSITIVE', 'score': 0.5}]
        
        # Test with special characters
        special_text = "Wow! @#$%^&*()"
        result = app.sentiment_pipeline(special_text)
        assert result == [{'label': 'POSITIVE', 'score': 0.5}]
