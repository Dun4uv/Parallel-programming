import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
EXECUTABLE = "./matrix_mul.exe"
DATA_DIR = "data"
RESULTS_DIR = "results"
VERIFY_SCRIPT = "verify/verify.py"

def generate_matrix(filename, n):
    matrix = np.random.uniform(0, 10, size=(n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(map(lambda x: f"{x:.4f}", row)) + "\n")

def run_test(n):
    file_a = os.path.join(DATA_DIR, "A.txt")
    file_b = os.path.join(DATA_DIR, "B.txt")
    file_res = os.path.join(RESULTS_DIR, "Result.txt")

    print(f"--- Тестирование N = {n} ---")
    
    generate_matrix(file_a, n)
    generate_matrix(file_b, n)
    
    cmd = [EXECUTABLE, file_a, file_b, file_res]
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    end_time = time.time()

    if result.returncode != 0:
        print(f"Ошибка при выполнении C++: {result.stderr}")
        return None


    exec_time = end_time - start_time 
    for line in result.stdout.split('\n'):
        if "Время:" in line:
            try:
                exec_time = float(line.split(":")[1].strip().split()[0])
            except: pass

    verify_cmd = ["python", VERIFY_SCRIPT, file_a, file_b, file_res]
    v_result = subprocess.run(verify_cmd, capture_output=True, text=True, encoding='utf-8')
    is_valid = "Все верно!" in v_result.stdout

    print(f"Время: {exec_time:.4f} сек | Валидация: {'OK' if is_valid else 'FAIL'}")
    return exec_time

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_data = []

    for n in SIZES:
        elapsed = run_test(n)
        if elapsed is not None:
            results_data.append((n, elapsed))

    ns = [r[0] for r in results_data]
    ts = [r[1] for r in results_data]

    plt.figure(figsize=(10, 6))
    plt.plot(ns, ts, marker='o', color='b', linestyle='-')
    plt.title('Зависимость времени умножения от размера матрицы')
    plt.xlabel('Размер матрицы N (NxN)')
    plt.ylabel('Время (секунды)')
    plt.grid(True)
    plt.savefig('performance_graph.png')
    print("\nЭксперименты завершены. График сохранен как performance_graph.png")

    print("\nТаблица:")
    print("| N | Время (сек) | Операций (2N³) |")
    print("|---|-------------|----------------|")
    for n, t in results_data:
        ops = 2 * (n**3)
        print(f"| {n} | {t:.4f} | {ops:,} |")

if __name__ == "__main__":
    main()