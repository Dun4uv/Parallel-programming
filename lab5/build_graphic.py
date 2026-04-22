import os
import re
import matplotlib.pyplot as plt

LOGS_DIR = "logs"


def parse_logs():
    results = []
    n_pattern = re.compile(r"Matrix:\s+(\d+)x")
    t_pattern = re.compile(r"Time:\s+([\d\.]+)")
    p_pattern = re.compile(r"Processes MPI:\s+(\d+)")

    if not os.path.exists(LOGS_DIR):
        print(f"Ошибка: Папка {LOGS_DIR} не найдена!")
        return results

    for filename in os.listdir(LOGS_DIR):
        if filename.endswith(".out"):
            with open(os.path.join(LOGS_DIR, filename), 'r') as f:
                content = f.read()
                n_match = n_pattern.search(content)
                t_match = t_pattern.search(content)
                p_match = p_pattern.search(content)

                if n_match and t_match and p_match:
                    results.append({
                        'N': int(n_match.group(1)),
                        'Time': float(t_match.group(1)),
                        'P': int(p_match.group(1))
                    })
    return results


def create_plot(data):
    if not data:
        print("Данные для построения не найдены.")
        return

    processes = sorted(list(set(d['P'] for d in data)))

    plt.figure(figsize=(10, 6))
    for p in processes:
        p_data = sorted([d for d in data if d['P'] == p], key=lambda x: x['N'])
        ns = [d['N'] for d in p_data]
        ts = [d['Time'] for d in p_data]

        plt.plot(ns, ts, marker='o', label=f'{p} процессов')

    plt.title('Зависимость времени выполнения от размера матрицы')
    plt.xlabel('Размер матрицы N (NxN)')
    plt.ylabel('Время (сек)')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_results.png')
    print("График производительности сохранен: performance_results.png")
    plt.show()


if __name__ == "__main__":
    log_data = parse_logs()
    create_plot(log_data)