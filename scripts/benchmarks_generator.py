import subprocess
import json
import os

parameters_init = list(range(1, 250, 20))
parameters_after = list(range(261, 1200, 50))
exe_files = ['./build/particle_sim', './build/particle_sim_fast']

def calculating():
    for exe_file in exe_files:
        results = {}
        for param in parameters_init:
            process = subprocess.Popen([exe_file, str(param), '400'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in process.stdout:
                print(line.strip())
            process.wait()

            last_line = line.strip()
            time_number = int(last_line.split()[-1])
            results[str(param)] = time_number

            print(results)

        for param in parameters_after:
            process = subprocess.Popen([exe_file, str(param), '400'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in process.stdout:
                print(line.strip())
            process.wait()

            last_line = line.strip()
            time_number = int(last_line.split()[-1])
            results[str(param)] = time_number

            print(results)
            
        json_name = os.path.splitext(os.path.basename(exe_file))[0]

        with open(f'{json_name}.json', 'w') as f:
            json.dump(results, f)
        print(f"----------Results saved to {json_name}.json----------")

calculating()
print("TESTING IS FINISHED ARE CALCULATED")
