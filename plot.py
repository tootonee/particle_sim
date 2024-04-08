import json
import matplotlib as mpl
import matplotlib.pyplot as plt

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def plot_lines(line1_data, line2_data, output_file = "comparison_two_lines.svg"):
    plt.figure(figsize=(10, 5))

    x_values = list(map(int, line1_data.keys()))
    y_values1 = list(line1_data.values())
    y_values2 = list(line2_data.values())

    plt.plot(x_values, y_values1, label='Line 1')
    plt.plot(x_values, y_values2, label='Line 2')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Lines from JSON Data')
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
