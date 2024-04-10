import json
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 45})

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def plot_lines(line1_data, line2_data, output_file = "rng_comp.svg"):
    plt.figure(figsize=(32, 18))

    x_values = list(map(int, line1_data.keys()))
    y_values1 = list(line1_data.values())
    y_values2 = list(line2_data.values())

    plt.plot(x_values, y_values1, label='Host RNG')
    plt.plot(x_values, y_values2, label='Device RNG')

    plt.xlabel('Amount of particle moves per iteration ~ amount of floats generated')
    plt.ylabel('Time to run 10, 000 iterations')
    plt.title('Comparison of RNG efficacy')
    plt.legend()
    plt.grid(True)

    plt.savefig(output_file)

    plt.show()

def main():
    json_file1 = 'particle_sim.json'
    json_file2 = 'particle_sim_fast.json'

    line1_data = read_json(json_file1)
    line2_data = read_json(json_file2)

    print(line1_data)
    print(line2_data)
    plot_lines(line1_data, line2_data)

if __name__ == "__main__":
    main()
