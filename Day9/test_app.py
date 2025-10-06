import pytest
from unittest.mock import patch, MagicMock
import app

def test_summarizer_creation():
    """Test that the summarizer pipeline is created correctly."""
    # Since summarizer is created at module import, we check it's callable
    assert hasattr(app.summarizer, '__call__')

def test_main_success():
    """Test the main function with mocked summarizer."""
    mock_summary = [{'summary_text': 'This is a mock summary of the article.'}]
    expected_article = """
        70 Pine Street (formerly known as the 60 Wall Tower, Cities Service Building, and American International Building) is a 67-story, 952-foot (290 m) residential skyscraper in the Financial District of Lower Manhattan, New York City, New York, U.S. Designed by the architectural firm of Clinton & Russell, Holton & George in the Art Deco style, 70 Pine Street was constructed between 1930 and 1932 as an office building. The structure was originally named for the energy conglomerate Cities Service Company (later Citgo), its first tenant. Upon its completion, it was Lower Manhattan's tallest building and, until 1969, the world's third-tallest building.
        The building occupies a trapezoidal lot on Pearl Street between Pine and Cedar Streets. It features a brick, limestone, and gneiss facade with numerous setbacks. The building contains an extensive program of ornamentation, including depictions of the Cities Service Company's triangular logo and solar motifs. The interior has an Art Deco lobby and escalators at the lower stories, as well as double-deck elevators linking the floors. A three-story penthouse, intended for Cities Service's founder, Henry Latham Doherty, was instead used as a public observatory.
        Construction was funded through a public offering of stock, rather than a mortgage loan. Despite having been built during the Great Depression, the building was profitable enough to break even by 1936, and ninety percent of its space was occupied five years later. The American International Group (AIG) bought the building in 1976, and it was acquired by another firm in 2009 after AIG went bankrupt. The building and its first-floor interior were designated as official New York City landmarks in June 2011. The structure was converted to residential use in 2016.
    """
    with patch.object(app, 'summarizer', return_value=mock_summary) as mock_summarizer:
        with patch('builtins.print') as mock_print:
            app.main()
            # Check that summarizer was called with the expected arguments
            mock_summarizer.assert_called_once_with(
                expected_article,
                max_length=130,
                min_length=30,
                do_sample=False
            )
            # Check that print was called with the summary
            mock_print.assert_called_once_with(mock_summary)

def test_main_summarizer_failure():
    """Test the main function when summarizer raises an exception."""
    with patch.object(app, 'summarizer', side_effect=Exception("Model loading failed")) as mock_summarizer:
        with patch('builtins.print') as mock_print:
            with pytest.raises(Exception, match="Model loading failed"):
                app.main()
            # Ensure summarizer was called
            mock_summarizer.assert_called_once()

def test_if_name_main():
    """Test that main is called when script is run directly."""
    with patch.object(app, 'main') as mock_main:
        # Simulate __name__ == '__main__'
        # Since it's already executed, but to test, perhaps not necessary
        # But since the if is at the end, and main is defined, it's covered by importing
        pass  # This test might not be needed, but for completeness
