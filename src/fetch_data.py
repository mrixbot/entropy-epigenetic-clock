import os
import urllib.request

def download_chan_matrices(download_dir="data/raw"):
    """
    Downloads the pre-calculated average methylation and entropy matrices 
    for GSE288139 directly from NCBI GEO supplementary files.
    """
    # Ensure the target directory exists at the root of the project
    os.makedirs(download_dir, exist_ok=True)
    
    # NCBI GEO download URL structure for specific supplementary files
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE288139&format=file&file="
    
    files_to_download = [
        "GSE288139_buccalswab_avgmeth.txt.gz",
        "GSE288139_buccalswab_entropy.txt.gz"
    ]
    
    print("[*] Initiating download for Chan et al. (GSE288139) matrices...")
    
    for filename in files_to_download:
        url = base_url + filename
        save_path = os.path.join(download_dir, filename)
        
        print(f"[*] Fetching {filename} (~{ '660' if 'avgmeth' in filename else '356' } Kb)...")
        try:
            # Download the file directly
            urllib.request.urlretrieve(url, save_path)
            print(f"[+] Successfully saved to {save_path}")
        except Exception as e:
            print(f"[!] Error downloading {filename}: {e}")
            print(f"[!] Target URL was: {url}")

    print("\n[+] Data acquisition complete! Ready for processing.")

if __name__ == "__main__":
    download_chan_matrices()