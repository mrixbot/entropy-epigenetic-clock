import urllib.request
import gzip
import os
import pandas as pd

def fetch_and_map_genes():
    """Abandoned attempt to pull gene symbols out of the GEO series matrix.

    The series matrix header doesn't carry per-locus annotation, so the parse loop
    below does nothing and the mapping is left unwritten. map_genes.py does this
    properly against Ensembl instead -- keep using that one.
    """
    print("==================================================")
    print(" NCBI GEO: FETCHING CUSTOM ANNOTATION DATA")
    print("==================================================")

    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE288nnn/GSE288139/matrix/GSE288139_series_matrix.txt.gz"
    zip_path = "GSE288139_series_matrix.txt.gz"

    if not os.path.exists(zip_path):
        print(f"[*] Downloading annotation data from GEO...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("[+] Download complete!")
        except Exception as e:
            print(f"[!] Failed to download. The dataset might rely on a separate GPL file. Error: {e}")
            return

    print("[*] Parsing GEO Matrix for Gene Symbols...")
    locus_to_gene = {}
    with gzip.open(zip_path, 'rt') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
            # never fills locus_to_gene -- the header has no annotation column
            if line.startswith('!Sample_characteristics_ch1') or 'gene' in line.lower():
                pass

    print("[*] Note: If the series matrix lacks the annotation, we will pull the attached GPL (Platform) file next.")

    top_cpgs_file = "outputs/top_entropy_cpgs.csv"
    if not os.path.exists(top_cpgs_file):
        print(f"[!] Cannot find {top_cpgs_file}. Did you run extract_pathways.py first?")
        return

    top_df = pd.read_csv(top_cpgs_file)

    # never got this far; superseded by map_genes.py
    # top_df['Gene_Symbol'] = top_df['CpG_Locus'].map(locus_to_gene)
    # top_df.to_csv("outputs/mapped_entropy_genes.csv", index=False)

    print("[+] Framework ready. Check the downloaded matrix file to confirm the annotation column headers!")

if __name__ == "__main__":
    fetch_and_map_genes()
