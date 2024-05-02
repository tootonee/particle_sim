import subprocess
import json
import os
import tqdm

params = list(range(100, 14101, 500))
exe_files = ['./build/particle_sim_device']
# exe_files = ['./build/particle_sim']

def calculating():
    for exe_file in exe_files:
        results = {}
        for param in tqdm.tqdm(params):
            lst = []
            for x in range(4):
                process = subprocess.Popen([exe_file, '60', str(param)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in process.stdout:
                    print(line.strip())
                process.wait()

                last_line = line.strip()
                time_number = int(last_line.split()[-1])
                lst.append(time_number)

            results[str(param)] = sum(lst) / len(lst)

            print(results)
            
        json_name = os.path.splitext(os.path.basename(exe_file))[0]

        with open(f'{json_name}.json', 'w') as f:
            json.dump(results, f)
        print(f"----------Results saved to {json_name}.json----------")

calculating()
print("TESTING IS FINISHED ARE CALCULATED")
