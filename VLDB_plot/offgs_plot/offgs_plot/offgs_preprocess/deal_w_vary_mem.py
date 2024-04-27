import csv

file_path = '/home/ubuntu/offgs_plot/offgs_plot/offgs_data/vary_cache.csv'

with open(file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    data = list(reader)

# Skipping the header row and ensuring all data is float or int as needed
data = [[float(cell) if cell.isdigit() else cell for cell in row] for row in data[1:]]

config = []
statistics = []
all_configs = []
all_statistics = []
all_normalized_statistics = []
for i, row in enumerate(data):
    if i % 3 == 0:
        config = []
        statistics = []
        config.append(row[0])
        config.append(row[1])
        statistics.append(row[-2])
    if i % 3 == 1:

        statistics.append(float(row[-2]) + float(row[-1]))  # Assumes row[-2] and row[-1] are now numbers
        statistics.append(row[-1])
    if i % 3 == 2:

        statistics.append(row[-1])
        statistics[1], statistics[2], statistics[3] = statistics[3], statistics[2], statistics[1]
        statistics = [float(stat) for stat in statistics]
        statistics_normalized = [round(stat / statistics[0], 2) for stat in statistics]
        all_configs.append(config)
        all_statistics.append(statistics)
        all_normalized_statistics.append(statistics_normalized)
        print(config)
        print(statistics)
        print(statistics_normalized)
