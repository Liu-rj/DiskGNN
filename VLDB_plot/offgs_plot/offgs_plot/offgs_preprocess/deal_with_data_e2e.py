import csv

file_path = '/home/ubuntu/offgs_plot/offgs_plot/Graph_NN_Benchmarks.csv'

with open(file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    data = list(reader)

# 修改数据：将第六列的值加到第五列
for row in data[1:]:  
    row[4] = str(round(float(row[4]) + float(row[5]),2)  )  
    row[4], row[5], row[6] = row[6], row[5], row[4]


# 打印修改后的数据
for row in data:
    print(",".join(row))

## save the new data
new_file_path = '/home/ubuntu/offgs_plot/offgs_plot/Graph_NN_Benchmarks_new.csv'
with open(new_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

for row in data[1:]:  
    normalize = float(row[2])
    for j in range(2, 7):
        row[j] = str(round(float(row[j])/normalize, 2))
## 
print('after normalization')
for row in data:
    
    print(",".join(row))
## save
new_file_path = '/home/ubuntu/offgs_plot/offgs_plot/Graph_NN_Benchmarks_new_normalize.csv'
with  open(new_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
