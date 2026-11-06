import urllib.request
import gzip
import pandas as pd
import re
import os


def fetch_and_clean_ages():
    """Pull patient ages out of the GSE288139 series matrix header."""
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
    # Keyed by column position rather than appended to a list: if one sample's age
    # field fails to parse, appending would shift every later age by one and quietly
    # attach the wrong age to the wrong patient.
    ages_by_position = {}

    for line in lines:
        if line.startswith('!Sample_title'):
            raw_titles = line.strip().split('\t')[1:]
            sample_ids = [x.strip('"').replace('Buccal_', '') for x in raw_titles]

        elif line.startswith('!Sample_characteristics_ch1') and 'age' in line.lower():
            raw_chars = line.strip().split('\t')[1:]
            for position, char in enumerate(raw_chars):
                if 'age' in char.lower():
                    # \d+ alone truncated 7.2 to "7". Ages here run 7.2-84.4 so the
                    # decimal matters.
                    match = re.search(r'\d+\.?\d*', char)
                    if match:
                        ages_by_position[position] = float(match.group())

    if not sample_ids or not ages_by_position:
        print("[!] Could not parse the metadata correctly. The formatting might be non-standard.")
        return

    # NaN for anything that didn't parse, so the gap stays visible
    ages = [ages_by_position.get(i, float('nan')) for i in range(len(sample_ids))]

    df = pd.DataFrame({'Age': ages}, index=sample_ids)
    df.index.name = 'Patient_ID'

    n_missing = df['Age'].isna().sum()
    if n_missing > 0:
        print(f"[!] {n_missing} sample(s) had no parseable age field -- these will show as NaN "
              f"and should be investigated, not silently dropped or misaligned:")
        print(df[df['Age'].isna()])

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean_ages.csv")

    print(f"[+] Successfully extracted ages for {df['Age'].notna().sum()} of {len(df)} patients "
          f"(decimals now preserved, e.g. 7.2 not 7).")
    print(f"[*] Patient Age Range: {df['Age'].min()} to {df['Age'].max()} years")
    print("[+] File saved to data/processed/clean_ages.csv")


if __name__ == "__main__":
    fetch_and_clean_ages()
