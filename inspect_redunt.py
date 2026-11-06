# Scratch script: dump the first few rows after the table marker so the column
# layout of the series matrix can be eyeballed. Not part of the pipeline.
with open('GSE288139_series_matrix.txt', 'r') as f:
    print("--- SEARCHING FOR TABLE HEADER ---")
    for line in f:
        if line.startswith('!series_matrix_table_begin'):
            # next(f) walks on from wherever the for loop left off
            for _ in range(4):
                print(next(f).strip()[:200])
            break
