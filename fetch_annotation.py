import urllib.request
import gzip
import os
import pandas as pd

def fetch_and_map_genes():
    print("==================================================")
    print(" NCBI GEO: FETCHING CUSTOM ANNOTATION DATA")
    print("==================================================")
    
    # 1. NCBI GEO Series Matrix FTP URL for GSE288139
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE288nnn/GSE288139/matrix/GSE288139_series_matrix.txt.gz"
    zip_path = "GSE288139_series_matrix.txt.gz"
    
    # 2. Download the matrix file
    if not os.path.exists(zip_path):
        print(f"[*] Downloading annotation data from GEO...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("[+] Download complete!")
        except Exception as e:
            print(f"[!] Failed to download. The dataset might rely on a separate GPL file. Error: {e}")
            return

    # 3. Parse the matrix to extract the annotation dictionary
    print("[*] Parsing GEO Matrix for Gene Symbols...")
    locus_to_gene = {}
    with gzip.open(zip_path, 'rt') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
            # Look for feature data lines in the GEO header
            if line.startswith('!Sample_characteristics_ch1') or 'gene' in line.lower():
                # This is a simplified extraction; we can refine this based on the exact GEO formatting
                pass 
                
    print("[*] Note: If the series matrix lacks the annotation, we will pull the attached GPL (Platform) file next.")
    
    # 4. Load your top CpG loci
    top_cpgs_file = "outputs/top_entropy_cpgs.csv"
    if not os.path.exists(top_cpgs_file):
        print(f"[!] Cannot find {top_cpgs_file}. Did you run extract_pathways.py first?")
        return
        
    top_df = pd.read_csv(top_cpgs_file)
    
    # Placeholder for mapping logic once the exact GPL format is downloaded
    # top_df['Gene_Symbol'] = top_df['CpG_Locus'].map(locus_to_gene)
    # top_df.to_csv("outputs/mapped_entropy_genes.csv", index=False)
    
    print("[+] Framework ready. Check the downloaded matrix file to confirm the annotation column headers!")

if __name__ == "__main__":
    fetch_and_map_genes()