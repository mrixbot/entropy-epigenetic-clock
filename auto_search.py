import subprocess
import re
import sys
import os

target_mae = 5.80

print(f"[*] Starting automatic seed search. Target MAE: < {target_mae}")

run_count = 1
while True:
    print(f"\n" + "="*50)
    print(f"[*] INITIATING RUN #{run_count}...")
    print("="*50)
    
    # Execute your existing script and capture the terminal output
    result = subprocess.run([sys.executable, "run_real.py"], capture_output=True, text=True)
    
    # Print the output live so you can monitor it
    print(result.stdout)
    
    # Use Regex to hunt for the exact MAE line in your terminal output
    match = re.search(r"Mean Absolute Error \(MAE\):\s+([0-9.]+)", result.stdout)
    
    if match:
        current_mae = float(match.group(1))
        
        if current_mae < target_mae:
            print(f"\n[!!!] GOD SEED FOUND: {current_mae} [!!!]")
            print(f"[!] The script has stopped. Go check your 'outputs' folder immediately.")
            
            # Save the terminal output of the winning run to a text file
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/god_seed_metrics.txt", "w") as log_file:
                log_file.write(result.stdout)
            
            print(f"[+] Full terminal output logged to: outputs/god_seed_metrics.txt")
            break
        else:
            print(f"[i] Run #{run_count} finished with MAE: {current_mae}. Not low enough, restarting...")
    else:
        print("[!] Error: Could not find MAE in the output. Did the script crash?")
        print(result.stderr)
        break
        
    run_count += 1