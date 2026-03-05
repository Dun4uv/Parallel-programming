import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
THREADS = 8
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

    print(f"--- Тестирование N = {n}, Потоков = {THREADS} ---")
    generate_matrix(file_a, n)
    generate_matrix(file_b, n)
    
    cmd = [EXECUTABLE, file_a, file_b, file_res, str(THREADS)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print(f"Ошибка: {result.stderr}")
            return None

        exec_time = 0.0
        for line in result.stdout.split('\n'):
            if "Время:" in line:
                exec_time = float(line.split(":")[1].strip().split()[0])

        verify_cmd = ["python", VERIFY_SCRIPT, file_a, file_b, file_res]
        v_result = subprocess.run(verify_cmd, capture_output=True, text=True)
        is_valid = "Все верно!" in v_result.stdout

        print(f"Время: {exec_time:.4f} сек | Валидация: {'OK' if is_valid else 'FAIL'}")
        return exec_time
    except Exception as e:
        print(f"Сбой при запуске: {e}")
        return None

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_data = []

    for n in SIZES:
        elapsed = run_test(n)
        if elapsed is not None:
            results_data.append((n, elapsed))

    if results_data:
        ns, ts = zip(*results_data)
        plt.figure(figsize=(10, 6))
        plt.plot(ns, ts, marker='o', linestyle='-', color='red', label=f'OpenMP ({THREADS} потока)')
        plt.title('Производительность матричного умножения (OpenMP)')
        plt.xlabel('Размер матрицы N')
        plt.ylabel('Время выполнения (сек)')
        plt.grid(True)
        plt.legend()
        plt.savefig('performance_results.png')
        print("\nГрафик сохранен в performance_results.png")

        print("\nТаблица:")
        print("| N | Время (сек) | Операций (2N³) | Потоков |")
        print("|---|-------------|----------------|---------|")
        for n, t in results_data:
            print(f"| {n} | {t:.4f} | {2*(n**3):,} | {THREADS} |")

if __name__ == "__main__":
    main()