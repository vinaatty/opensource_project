import matplotlib.pyplot as plt

# 파일에서 실행 시간 데이터 읽기
def read_times(filename):
    N_values = []
    times = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:  # "소요시간 (N=1000): 0.000123 초"
                N = int(parts[2].split('=')[1][:-1])  # N=1000에서 숫자 추출
                time = float(parts[4])  # 수행 시간 추출
                N_values.append(N)
                times.append(time)
    return N_values, times

# CPU 및 GPU 데이터 불러오기
cpu_N, cpu_times = read_times("cpu_times.txt")
gpu_N, gpu_times = read_times("gpu_times.txt")

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(cpu_N, cpu_times, marker='o', linestyle='-', color='red', label="CPU")
plt.plot(gpu_N, gpu_times, marker='s', linestyle='--', color='blue', label="GPU")

plt.xlabel("N (데이터 크기)")
plt.ylabel("수행 시간 (초)")
plt.title("CPU vs GPU 수행 시간 비교")
plt.legend()
plt.grid(True)
plt.show()
