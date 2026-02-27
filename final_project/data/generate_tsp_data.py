"""
TSP 数据生成脚本 — 生成 DIFUSCO 兼容格式的 TSP 实例

输出格式（每行一个实例）：
  x1 y1 x2 y2 ... xN yN output t1 t2 ... tN t1

坐标为 [0,1] 范围浮点数，tour 为 1-indexed 城市编号（含回到起点）

用法示例：
  python data/generate_tsp_data.py --num_nodes 20 --num_samples 1000 --output_file data/tsp20_train.txt
  python data/generate_tsp_data.py --num_nodes 50 --num_samples 5000 --output_file data/tsp50_train.txt --solver elkai
"""

import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


# ============================================================
# 距离矩阵
# ============================================================

def compute_distance_matrix(coords):
    """计算欧氏距离矩阵 (N, N)"""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


# ============================================================
# TSP 求解器
# ============================================================

def solve_tsp_elkai(coords):
    """
    使用 elkai (LKH-3 绑定) 求解。
    coords: (N, 2) numpy array, 值在 [0, 1]
    返回: 0-indexed tour list, 长度 N（不含回到起点）
    """
    import elkai
    SCALE = 1_000_000
    dist_matrix = compute_distance_matrix(coords)
    int_dist = (dist_matrix * SCALE).astype(int).tolist()
    cities = elkai.DistanceMatrix(int_dist)
    tour = cities.solve_tsp()  # 返回 [0, ..., 0]
    return tour[:-1]  # 去掉末尾回到起点


def solve_tsp_python_tsp(coords, method='lk'):
    """
    使用 python-tsp 求解（备用方案）。
    method: 'dp' (精确, N<=20), 'lk' (LK启发式), 'sa' (模拟退火)
    """
    dist_matrix = compute_distance_matrix(coords)
    if method == 'dp':
        from python_tsp.exact import solve_tsp_dynamic_programming
        permutation, _ = solve_tsp_dynamic_programming(dist_matrix)
    elif method == 'lk':
        from python_tsp.heuristics import solve_tsp_lin_kernighan
        permutation, _ = solve_tsp_lin_kernighan(dist_matrix)
    elif method == 'sa':
        from python_tsp.heuristics import solve_tsp_simulated_annealing
        permutation, _ = solve_tsp_simulated_annealing(dist_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    return list(permutation)


# ============================================================
# 数据格式化
# ============================================================

def format_instance(coords, tour):
    """
    格式化为 DIFUSCO 兼容的单行字符串。
    coords: (N, 2), tour: 0-indexed list of length N
    输出: "x1 y1 x2 y2 ... xN yN output t1 t2 ... tN t1"
    """
    # 坐标部分
    coord_strs = []
    for x, y in coords:
        coord_strs.append(f"{x:.8f}")
        coord_strs.append(f"{y:.8f}")

    # tour 部分（转为 1-indexed，末尾回到起点）
    tour_1indexed = [str(t + 1) for t in tour]
    tour_1indexed.append(str(tour[0] + 1))

    return " ".join(coord_strs) + " output " + " ".join(tour_1indexed)


# ============================================================
# 单实例生成（供多进程调用）
# ============================================================

def generate_single_instance(idx, num_nodes, solver, seed_base):
    """生成单个 TSP 实例，返回格式化字符串。"""
    rng = np.random.RandomState(seed_base + idx)
    coords = rng.uniform(0, 1, size=(num_nodes, 2))

    if solver == 'elkai':
        tour = solve_tsp_elkai(coords)
    elif solver == 'python-tsp-lk':
        tour = solve_tsp_python_tsp(coords, method='lk')
    elif solver == 'python-tsp-dp':
        tour = solve_tsp_python_tsp(coords, method='dp')
    elif solver == 'python-tsp-sa':
        tour = solve_tsp_python_tsp(coords, method='sa')
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return format_instance(coords, tour)


# ============================================================
# 验证
# ============================================================

def validate_line(line):
    """验证单行数据格式正确，返回 (is_valid, error_msg)"""
    parts = line.strip().split(" output ")
    if len(parts) != 2:
        return False, "missing 'output' separator"

    coords_vals = list(map(float, parts[0].split()))
    tour_vals = list(map(int, parts[1].split()))
    N = len(coords_vals) // 2

    if len(coords_vals) % 2 != 0:
        return False, "odd number of coordinate values"
    if len(tour_vals) != N + 1:
        return False, f"tour length {len(tour_vals)}, expected {N + 1}"
    if tour_vals[0] != tour_vals[-1]:
        return False, "tour doesn't return to start"
    if set(tour_vals[:-1]) != set(range(1, N + 1)):
        return False, "tour doesn't visit all cities exactly once"
    return True, ""


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate TSP dataset (DIFUSCO format)')
    parser.add_argument('--num_nodes', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--solver', type=str, default='elkai',
                        choices=['elkai', 'python-tsp-lk', 'python-tsp-dp', 'python-tsp-sa'])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    print(f"Generating {args.num_samples} TSP-{args.num_nodes} instances "
          f"using {args.solver} solver...")

    worker_fn = partial(
        generate_single_instance,
        num_nodes=args.num_nodes,
        solver=args.solver,
        seed_base=args.seed,
    )

    results = []
    with Pool(processes=args.num_workers) as pool:
        for line in tqdm(pool.imap(worker_fn, range(args.num_samples)),
                         total=args.num_samples, desc=f"TSP-{args.num_nodes}"):
            results.append(line)

    # 写入文件
    with open(args.output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')

    print(f"Saved {len(results)} instances to {args.output_file}")

    # 验证前 5 条
    errors = 0
    for i, line in enumerate(results[:5]):
        ok, msg = validate_line(line)
        if not ok:
            print(f"  [FAIL] Line {i}: {msg}")
            errors += 1
    if errors == 0:
        print("Validation: first 5 lines OK")


if __name__ == '__main__':
    main()
