import argparse

from data_extractor.sinan_extractor import SinanDataExtractor

def main():
    parser = argparse.ArgumentParser(description="SINAN Data Extractor")
    parser.add_argument('--path', type=str, required=False, help='Path to save the data')
    parser.add_argument('--disease-codes', type=str, nargs='+', required=False, help='List of disease codes to extract')
    parser.add_argument('--frequency', type=str, required=False, help='Frequency of data extraction (default: MS)')

    args = parser.parse_args()

    extractor = SinanDataExtractor(
        path=args.path, 
        disease_codes=args.disease_codes, 
        frequency=args.frequency, 
    )
    extractor.extract()

if __name__ == "__main__":
    main()