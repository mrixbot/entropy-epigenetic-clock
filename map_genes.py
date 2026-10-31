import csv
import urllib.request
import json
import time

def map_coordinates_to_genes(csv_path="outputs/top_entropy_cpgs.csv"):
    print("[*] Querying Ensembl for gene annotations using built-in libraries...")
    
    with open(csv_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        
    for row in rows:
        locus = row['CpG_Locus']
        try:
            chrom, pos_range = locus.split(':')
            start, end = pos_range.split('-')
            
            url = f"https://rest.ensembl.org/overlap/region/human/{chrom}:{start}-{end}?feature=gene"
            
            req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    if data and isinstance(data, list):
                        symbols = list(set([gene.get('external_name', 'Unknown') for gene in data if 'external_name' in gene]))
                        row['Gene_Symbol'] = ",".join(symbols) if symbols else "Intergenic"
                    else:
                        row['Gene_Symbol'] = "Intergenic"
                else:
                    row['Gene_Symbol'] = f"API_Error_{response.status}"
            
            time.sleep(0.1) 
        except Exception as e:
            row['Gene_Symbol'] = "Parse_Error"
            
    if rows:
        fieldnames = list(rows[0].keys())
        if 'Gene_Symbol' not in fieldnames:
            fieldnames.append('Gene_Symbol')
            
        with open("outputs/mapped_entropy_genes.csv", mode='w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
    print("[+] Done! Mapped gene symbols saved to outputs/mapped_entropy_genes.csv")

if __name__ == "__main__":
    map_coordinates_to_genes()