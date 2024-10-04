import os
import pandas as pd

# 定义 FoldX 可执行文件的路径（你需要设置正确的路径）
foldx_path = '/home/hongtan/pretrain_single/code/foldx5_1Linux64_0/foldx'  # 替换为你FoldX的实际路径

# 读取CSV文件
file_path = '/home/hongtan/pretrain_single/code/foldx_cac/parsed_mutations.csv'  # 替换为你的实际文件路径
df = pd.read_csv(file_path)

# 获取每个PDB的唯一值
pdb_codes = df['pdbcode'].unique()

# 定义一个函数来创建突变文件
def create_mutation_file(mutations, filename='mutations.txt'):
    with open(filename, 'w') as f:
        for mutation in mutations:
            f.write(mutation + ";\n")  # 每个突变都写入文件

# 定义一个函数来调用FoldX并计算ΔΔG
def calculate_ddg(pdb_code, mutations):
    # 只修复一次PDB结构
    repair_command = f"{foldx_path} --command=RepairPDB --pdb={pdb_code}.pdb"
    os.system(repair_command)

    # 对每个突变进行计算
    ddg_results = []
    for mutation in mutations:
        # 创建突变文件
        create_mutation_file([mutation])

        # 计算突变的ΔΔG
        build_model_command = f"{foldx_path} --command=BuildModel --pdb={pdb_code}_Repair.pdb --mutant-file=mutations.txt"
        os.system(build_model_command)
        
        # 读取结果文件，获取ΔΔG值
        results_file = f"DifferenceEnergy_{pdb_code}_Repair.fxout"
        try:
            with open(results_file, 'r') as f:
                for line in f.readlines():
                    if line.startswith("Pdb Total"):
                        ddg = float(line.split()[3])  # ΔΔG通常在第4列
                        ddg_results.append([pdb_code, mutation, ddg])
        except FileNotFoundError:
            print(f"Result file {results_file} not found!")
            ddg_results.append([pdb_code, mutation, None])

    return ddg_results

# 创建一个空列表来存储所有的结果
all_results = []

# 对每个唯一的PDB文件进行一次修复，然后计算所有突变的ΔΔG
for pdb_code in pdb_codes:
    # 筛选与该PDB相关的突变
    mutations = df[df['pdbcode'] == pdb_code]['parsed_mutation'].tolist()
    
    # 计算与该PDB相关的所有突变的ΔΔG
    ddg_results = calculate_ddg(pdb_code.lower(), mutations)
    
    # 将结果合并到总结果中
    all_results.extend(ddg_results)

# 将所有结果保存到新的CSV文件中
output_df = pd.DataFrame(all_results, columns=['pdbcode', 'mutation', 'calculated_ddg'])
output_df.to_csv('./calculated_ddg_results_optimized.csv', index=False)

print("FoldX ΔΔG calculations completed. Results saved to calculated_ddg_results_optimized.csv.")
