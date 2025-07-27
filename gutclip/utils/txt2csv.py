import csv

input_file = '/data/home/wudezhi/project/school/x-meta/datasets/raw/tree/count_matrix_top1000_0715.txt'
output_file = '/data/home/wudezhi/project/school/x-meta/datasets/raw/tree/count_matrix_top1000_0715.tsv'

with open(input_file, 'r', encoding='utf-8') as fin:
    lines = fin.readlines()

# 第一行是物种名
species_names = lines[0].strip().split()
header = ['Sample'] + species_names  # 构建表头：Sample + 物种名

# 解析数据行（每行一个样本）
data = []
for line in lines[1:]:
    parts = line.strip().split()
    if not parts:
        continue
    sample = parts[0]
    abundances = parts[1:]
    data.append([sample] + abundances)

# 写入tsv文件
with open(output_file, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout, delimiter='\t')
    writer.writerow(header)
    writer.writerows(data)

print(f'转换完成，结果保存在 {output_file}')

"""
otu
"""

# import csv

# input_file = '/data/home/wudezhi/project/school/x-meta/datasets/raw/tree/top1000_otu.txt'
# output_file = '/data/home/wudezhi/project/school/x-meta/datasets/raw/tree/top1000_otu.csv'

# with open(input_file, 'r', encoding='utf-8') as fin, \
#      open(output_file, 'w', newline='', encoding='utf-8') as fout:
#     for line in fin:
#         # 自动按空白字符（空格或tab）分割
#         row = line.strip().split()
#         # 写入csv
#         fout.write(','.join(row) + '\n')

# print(f'转换完成，结果保存在 {output_file}')