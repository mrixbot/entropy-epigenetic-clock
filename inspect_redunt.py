with open('GSE288139_series_matrix.txt', 'r') as f:
    print("--- SEARCHING FOR TABLE HEADER ---")
    for line in f:
        if line.startswith('!series_matrix_table_begin'):
            for _ in range(4):
                print(next(f).strip()[:200])
            break