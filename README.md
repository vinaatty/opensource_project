#GPU 프로그래밍 
CPU vs GPU 보고서
1. N=10인 경우
![스크린샷 2025-03-25 16-06-51](https://github.com/user-attachments/assets/f679248c-95c3-45bb-b09f-972d665f9df8)
2. N=10000인 경우
![스크린샷 2025-03-25 16-08-16](https://github.com/user-attachments/assets/f904b37d-0e72-4999-9c74-92efd007caed)

3. N={1000, 2000,..., 10000} 인 경우 CPU 및 GPU 수행시간 결과 도표
   ![스크린샷 2025-04-03 15-14-00](https://github.com/user-attachments/assets/8db40118-a0fc-4a93-8d9d-3913538defb3)
![스크린샷 2025-04-03 15-13-10](https://github.com/user-attachments/assets/a1eacf9f-4cf0-43ce-8d6e-7fed84045da9)

  - nvcc add_loop_cpu.cu -o add_cpu 와 nvcc add_loop_gpu.cu add_gpu를 수행해서 컴파일
  - ./add.cpu 와 ./add.gpu 각각 실행하면 수행시간 출력
  - ./add_cpu > cpu_result.txt 와 ./add_gpu > gpu_result.txt 수행 후 paste cpu_result.txt gpu_result.txt | column -t 하면 정렬된 표 출력함
  - paste cpu_results.txt gpu_results.txt | awk '{printf "%-8s %-12s %-8s %-12s\n", $1, $2, $3, $4}' 이런식으로 하면 열 간격이 맞춰진 표 출력(필수 x)


5. N=10000 인 경우 N/2로 나누어 두번 Kernel 을 수행하도록 작성한 코드 및 수행시간 결과
   - book.h와 add_loop_cpu.cu 코드는 변화x(2의 코드와 같음). add_loop_gpu.cu 만 수정해서 출력.
   - Q.CPU 코드는 수정할 필요가 없는 이유?
     A: 
CPU 코드에서는 while 루프가 tid = 0에서 tid = N까지 모든 데이터를 한 번에 처리
GPU처럼 N/2로 나눠서 실행할 필요가 없음!
CPU는 순차적으로 처리하므로, 한 번의 while 루프에서 모든 연산이 완료됨
CPU 코드에서 N/2로 나누어 두 번 실행해야 할 필요가 없는 이유
CPU는 순차적으로 동작 → 기존 코드에서 한 번에 전체 데이터를 계산할 수 있음
CPU는 GPU처럼 많은 스레드를 활용하지 않음 → N/2로 나누어 실행해도 성능 차이가 없음
CPU에서 N/2씩 나눠서 실행해도 결국 같은 루프를 두 번 도는 것과 같음
즉, 기존 코드 그대로 실행하면 같은 결과가 나오고, 불필요한 작업이 추가되지 않음

GPU에서는 N/2로 나눠야 하는 이유
✅ GPU는 병렬 연산을 활용하는 방식
GPU는 스레드 블록(block)을 이용하여 데이터를 처리
원래 코드에서는 add<<<N, 1>>> 형태로 모든 데이터(N개)를 한 번에 연산
하지만 N이 클 경우, 커널을 나눠서 실행하는 것이 더 효율적일 수 있음
따라서 GPU에서는 N/2 크기의 두 개의 Kernel을 실행하여 병렬 처리

