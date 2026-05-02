import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
BLOCK_SIZES = [8, 16, 32]
EXECUTABLE = "./matrix_mul.exe"
DATA_DIR = "data"
RESULTS_DIR = "results"
VERIFY_SCRIPT = "verify/verify.py"


def generate_matrix(filename, n):
    """Генерирует случайную матрицу"""
    matrix = np.random.uniform(0, 10, size=(n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(map(lambda x: f"{x:.4f}", row)) + "\n")


def run_test(n, block_size, file_a, file_b, file_res):
    """Запускает программу и парсит вывод"""
    print(f"Тестирование: Размер блока = {block_size}x{block_size}")

    cmd = [EXECUTABLE, file_a, file_b, file_res, str(block_size)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

        if result.returncode != 0:
            print(f"Ошибка выполнения CUDA: {result.stderr}")
            return None

        exec_time = 0.0
        for line in result.stdout.split('\n'):
            if "Время:" in line:
                exec_time = float(line.split(":")[1].strip().split()[0])

        is_valid = False
        verify_cmd = ["python", VERIFY_SCRIPT, file_a, file_b, file_res]
        v_result = subprocess.run(verify_cmd, capture_output=True, text=True)
        is_valid = "Все верно!" in v_result.stdout
        valid_text = 'OK' if is_valid else 'FAIL'

        print(f"Время: {exec_time:.4f} сек | Валидация: {valid_text}")
        return exec_time

    except Exception as e:
        print(f"Сбой при запуске: {e}")
        return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    file_a = os.path.join(DATA_DIR, "A.txt")
    file_b = os.path.join(DATA_DIR, "B.txt")
    file_res = os.path.join(RESULTS_DIR, "Result.txt")

    results_by_block = {b: [] for b in BLOCK_SIZES}
    flat_results = []

    for n in SIZES:
        print(f"\n=== N = {n} (Генерация матриц...) ===")
        generate_matrix(file_a, n)
        generate_matrix(file_b, n)

        for b in BLOCK_SIZES:
            elapsed = run_test(n, b, file_a, file_b, file_res)
            if elapsed is not None:
                results_by_block[b].append((n, elapsed))
                flat_results.append((n, b, elapsed))

    if flat_results:
        plt.figure(figsize=(10, 6))
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, b in enumerate(BLOCK_SIZES):
            if results_by_block[b]:
                ns, ts = zip(*results_by_block[b])
                plt.plot(ns, ts, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Блок {b}x{b}')

        plt.title('Производительность матричного умножения (CUDA)')
        plt.xlabel('Размер матрицы N')
        plt.ylabel('Время выполнения ядра (сек)')
        plt.grid(True)
        plt.legend()
        plt.savefig('performance_results.png')
        print("\nГрафик сохранен в performance_results.png")

        print("\nТаблица результатов:")
        print("| N | Размер блока | Время (сек) | Операций (2N³) |")
        print("|---|--------------|-------------|----------------|")
        for n, b, elapsed in flat_results:
            print(f"| {n} | {b}x{b} | {elapsed:.4f} | {2 * (n ** 3):,} |")


if __name__ == "__main__":
    main()