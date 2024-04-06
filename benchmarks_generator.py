import subprocess
import json
import os

parameters = list(range(100, 1501, 100))
exe_files = ['./build/particle_sim', './build/particle_sim_fast']


for exe_file in exe_files:
    results = {}
    for param in parameters:
        process = subprocess.Popen([exe_file, '200', str(param)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            print(line.strip())
        process.wait()

        last_line = line.strip()
        time_number = int(last_line.split()[-1])
        results[str(param)] = time_number


    json_name = os.path.splitext(os.path.basename(exe_file))[0]

    with open(f'{json_name}.json', 'w') as f:
        json.dump(results, f)

print("Results saved to output.json")
