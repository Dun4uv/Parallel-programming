import os
import subprocess
import numpy as np
import time

SIZES = [400, 800, 1200, 1600, 2000]
PROCESSES = [1, 2, 4, 8, 16]
EXECUTABLE = "./matrixMulMPI"
DATA_DIR = "data"
RESULTS_DIR = "results"
LOGS_DIR = "logs"


def generate_matrix(filename, n):
    """Генерирует матрицу"""
    matrix = np.random.uniform(0, 10, size=(n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(map(lambda x: f"{x:.4f}", row)) + "\n")


def submit_job(n, p):
    """Создает .pbs файл и отправляет его через sbatch"""
    file_a = os.path.join(DATA_DIR, f"A_{n}.txt")
    file_b = os.path.join(DATA_DIR, f"B_{n}.txt")
    file_res = os.path.join(RESULTS_DIR, f"res_{n}_{p}.txt")
    log_file = os.path.join(LOGS_DIR, f"job_{n}_{p}.out")

    job_content = f"""#!/bin/bash
#SBATCH --job-name=mat_n{n}_p{p}
#SBATCH --time=0:05:00
#SBATCH --ntasks={p}
#SBATCH --partition=batch
#SBATCH --output={log_file}

module load intel/mpi4
mpirun {EXECUTABLE} {file_a} {file_b} {file_res}
"""

    script_name = f"temp_job_{n}_{p}.pbs"
    with open(script_name, "w") as f:
        f.write(job_content)

    result = subprocess.run(
        ["sbatch", script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if result.returncode == 0:
        print(f" Задача N={n}, P={p} отправлена. ID: {result.stdout.strip().split()[-1]}")
    else:
        print(f" Не удалось отправить N={n}, P={p}: {result.stderr}")

    os.remove(script_name)


def main():
    for d in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    for n in SIZES:
        print(f"\n=== Подготовка матриц для N = {n} ===")
        file_a = os.path.join(DATA_DIR, f"A_{n}.txt")
        file_b = os.path.join(DATA_DIR, f"B_{n}.txt")

        if not os.path.exists(file_a):
            generate_matrix(file_a, n)
            generate_matrix(file_b, n)

        for p in PROCESSES:
            submit_job(n, p)
            time.sleep(0.2)

    print("\nВсе задачи отправлены!")

if __name__ == "__main__":
    main()