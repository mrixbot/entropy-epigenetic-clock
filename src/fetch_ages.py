import urllib.request
import gzip
import pandas as pd
import re
import os

def fetch_and_clean_ages():
    # Standard NCBI FTP URL for GEO Series Matrices
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE288nnn/GSE288139/matrix/GSE288139_series_matrix.txt.gz"
    save_path = "data/raw/GSE288139_series_matrix.txt.gz"
    
    print("[*] Fetching clinical metadata (ages) from GEO FTP server...")
    try:
        urllib.request.urlretrieve(url, save_path)
    except Exception as e:
        print(f"[!] Failed to download: {e}")
        return
    
    print("[*] Parsing Series Matrix for patient ages...")
    with gzip.open(save_path, 'rt') as f:
        lines = f.readlines()
        
    sample_ids = []
    ages = []
    
    for line in lines:
        # Extract Patient IDs
        if line.startswith('!Sample_title'):
            raw_titles = line.strip().split('\t')[1:]
            # GEO titles look like "Buccal_P1111...". We strip "Buccal_" to match our clean matrices.
            sample_ids = [x.strip('"').replace('Buccal_', '') for x in raw_titles]
            
        # Extract Ages
        elif line.startswith('!Sample_characteristics_ch1') and 'age' in line.lower():
            raw_chars = line.strip().split('\t')[1:]
            for char in raw_chars:
                if 'age' in char.lower():
                    match = re.search(r'\d+', char)
                    if match:
                        ages.append(float(match.group()))
                        
    if not sample_ids or not ages:
        print("[!] Could not parse the metadata correctly. The formatting might be non-standard.")
        return
        
    # Construct final dataframe
    df = pd.DataFrame({'Age': ages}, index=sample_ids)
    df.index.name = 'Patient_ID'
    
    # Save the aligned target variables
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean_ages.csv")
    
    print(f"[+] Successfully extracted ages for {len(df)} patients!")
    print(f"[*] Patient Age Range: {df['Age'].min()} to {df['Age'].max()} years")
    print("[+] File saved to data/processed/clean_ages.csv")

if __name__ == "__main__":
    fetch_and_clean_ages()