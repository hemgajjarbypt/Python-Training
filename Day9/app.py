from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def main():
    article = """
        70 Pine Street (formerly known as the 60 Wall Tower, Cities Service Building, and American International Building) is a 67-story, 952-foot (290 m) residential skyscraper in the Financial District of Lower Manhattan, New York City, New York, U.S. Designed by the architectural firm of Clinton & Russell, Holton & George in the Art Deco style, 70 Pine Street was constructed between 1930 and 1932 as an office building. The structure was originally named for the energy conglomerate Cities Service Company (later Citgo), its first tenant. Upon its completion, it was Lower Manhattan's tallest building and, until 1969, the world's third-tallest building.
        The building occupies a trapezoidal lot on Pearl Street between Pine and Cedar Streets. It features a brick, limestone, and gneiss facade with numerous setbacks. The building contains an extensive program of ornamentation, including depictions of the Cities Service Company's triangular logo and solar motifs. The interior has an Art Deco lobby and escalators at the lower stories, as well as double-deck elevators linking the floors. A three-story penthouse, intended for Cities Service's founder, Henry Latham Doherty, was instead used as a public observatory.
        Construction was funded through a public offering of stock, rather than a mortgage loan. Despite having been built during the Great Depression, the building was profitable enough to break even by 1936, and ninety percent of its space was occupied five years later. The American International Group (AIG) bought the building in 1976, and it was acquired by another firm in 2009 after AIG went bankrupt. The building and its first-floor interior were designated as official New York City landmarks in June 2011. The structure was converted to residential use in 2016.
    """
    print(summarizer(article, max_length=130, min_length=30, do_sample=False))

if __name__ == '__main__':
    main()