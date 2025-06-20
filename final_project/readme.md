2. 설계 및 구현
기존 모델은 다음과 같은 구조로 구성된다:
입력: 28x28 흑백 이미지
Conv1 (20 filters, 5x5) + ReLU + MaxPool
Conv2 (50 filters, 5x5) + ReLU + MaxPool
FC1 (800→500) + ReLU + LRN
FC2 (500→10) + Softmax
2.2 개선 사항

valid_set_of_6 폴더: 15장의 6.pgm
인식은 됨 -> 7/15, (기존 1,3,5 도 잘 인식함)

시도1. 보정 가중치를 적용
- 6번클래스 확률을 약간 증폭함
  // 보정 가중치 적용
float six_bias = 1.20f;  // 6번 클래스 확률을 약간 증폭
result[6] = result[6] * six_bias;
시도 결과: 변화x

시도2. 같은 이미지를 여러 번(예: 5회) 약간씩 변형(노이즈 추가)해서 예측하고
다수결로 최종 클래스를 결정하는 방식으로 classify_example을 수정
시도 결과 : 변화 x

시도3. six_bias= 1.60f, 1.80f, 시도
1.80f : 
user@AIX-411-UBUNTU-31:~/GPU_yr/mnist_cudnn/mnistCUDNN$ ./mnistCUDNN
cudnnGetVersion() : 91000 , CUDNN_VERSION from cudnn.h : 91000 (9.10.0)
Host compiler version : GCC 11.4.0
There are 1 CUDA capable devices on your machine :
device 0 : sms 68  Capabilities 8.6, SmClock 1710.0 Mhz, MemSize (Mb) 10002, MemClock 9501.0 Mhz, Ecc=0, boardGroupID=0
Using device 0

Testing single precision
Loading image data/one_28x28.pgm (5 times)
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.026624 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.030720 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.051520 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.059168 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.061216 time requiring 184784 memory
Voting Result: 0 5 0 0 0 0 0 0 0 0 -> Final predict: 1
Loading image data/three_28x28.pgm (5 times)
Voting Result: 0 0 0 5 0 0 0 0 0 0 -> Final predict: 3
Loading image data/five_28x28.pgm (5 times)
Voting Result: 0 0 0 0 0 5 0 0 0 0 -> Final predict: 5

Result of classification: 1 3 5

Test passed!
Loading image ./valid_set_of_6/6_06.pgm (5 times)
Voting Result: 0 0 0 0 0 2 3 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_07.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_04.pgm (5 times)
Voting Result: 0 0 0 0 0 4 1 0 0 0 -> Final predict: 5
Predicted: 5
Loading image ./valid_set_of_6/6_05.pgm (5 times)
Voting Result: 0 0 1 0 0 0 0 0 4 0 -> Final predict: 8
Predicted: 8
Loading image ./valid_set_of_6/6_11.pgm (5 times)
Voting Result: 0 0 0 0 0 0 2 0 3 0 -> Final predict: 8
Predicted: 8
Loading image ./valid_set_of_6/6_15.pgm (5 times)
Voting Result: 0 0 0 0 0 4 1 0 0 0 -> Final predict: 5
Predicted: 5
Loading image ./valid_set_of_6/6_18.pgm (5 times)
Voting Result: 0 0 0 0 0 1 4 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_03.pgm (5 times)
Voting Result: 0 0 0 0 0 0 4 1 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_16.pgm (5 times)
Voting Result: 0 0 0 0 0 0 3 0 2 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_02.pgm (5 times)
Voting Result: 0 0 0 0 0 5 0 0 0 0 -> Final predict: 5
Predicted: 5
Loading image ./valid_set_of_6/6_10.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_01.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_14.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_17.pgm (5 times)
Voting Result: 0 0 0 0 0 5 0 0 0 0 -> Final predict: 5
Predicted: 5
Loading image ./valid_set_of_6/6_09.pgm (5 times)
Voting Result: 0 0 0 0 0 2 3 0 0 0 -> Final predict: 6
Predicted: 6
\n[6번 숫자 인식 결과] 9 / 15 성공 (60% 정확도)

Testing half precision (math in single precision)
Loading image data/one_28x28.pgm (5 times)
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.023552 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.027648 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.031456 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.053536 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.057344 time requiring 184784 memory
Voting Result: 0 5 0 0 0 0 0 0 0 0 -> Final predict: 1
Loading image data/three_28x28.pgm (5 times)
Voting Result: 0 0 0 5 0 0 0 0 0 0 -> Final predict: 3
Loading image data/five_28x28.pgm (5 times)
Voting Result: 0 0 0 0 0 5 0 0 0 0 -> Final predict: 5

Result of classification: 1 3 5

Test passed!
Loading image ./valid_set_of_6/6_06.pgm (5 times)
Voting Result: 0 0 0 0 0 5 0 0 0 0 -> Final predict: 5
Predicted: 5
Loading image ./valid_set_of_6/6_07.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_04.pgm (5 times)
Voting Result: 0 0 0 0 0 5 0 0 0 0 -> Final predict: 5
Predicted: 5
Loading image ./valid_set_of_6/6_05.pgm (5 times)
Voting Result: 0 0 0 0 0 0 0 0 5 0 -> Final predict: 8
Predicted: 8
Loading image ./valid_set_of_6/6_11.pgm (5 times)
Voting Result: 0 0 0 0 0 0 0 0 5 0 -> Final predict: 8
Predicted: 8
Loading image ./valid_set_of_6/6_15.pgm (5 times)
Voting Result: 0 0 0 0 0 5 0 0 0 0 -> Final predict: 5
Predicted: 5
Loading image ./valid_set_of_6/6_18.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_03.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_16.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_02.pgm (5 times)
Voting Result: 0 0 0 0 0 5 0 0 0 0 -> Final predict: 5
Predicted: 5
Loading image ./valid_set_of_6/6_10.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_01.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_14.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
Loading image ./valid_set_of_6/6_17.pgm (5 times)
Voting Result: 0 0 0 0 0 5 0 0 0 0 -> Final predict: 5
Predicted: 5
Loading image ./valid_set_of_6/6_09.pgm (5 times)
Voting Result: 0 0 0 0 0 0 5 0 0 0 -> Final predict: 6
Predicted: 6
\n[6번 숫자 인식 결과] 8 / 15 성공 (53.3333% 정확도)


시도5. + 노이즈 더 강하게, 밝기 추가

        // 노이즈 추가 (약간씩만)
        for (int idx = 0; idx < IMAGE_W*IMAGE_H; ++idx) {
            // 더 강한 노이즈: -0.04 ~ +0.04
            double noise = ((rand() % 81) - 40) / 1000.0; // -0.04 ~ +0.04
                // 밝기: -0.05 ~ +0.05, 컨트라스트: 0.85 ~ 1.15
            double alpha = 0.85 + (rand() % 31) / 100.0; // 0.85~1.15
            double beta = ((rand() % 11) - 5) / 100.0;   // -0.05~+0.05

            double v = static_cast<double>(imgData_h[idx]);
            v = v * alpha + beta + noise;
            if (v < 0.0) v = 0.0;
            if (v > 1.0) v = 1.0;
            imgData_h[idx] = static_cast<value_type>(v);

결과: \n[6번 숫자 인식 결과] 12 / 15 성공 (80% 정확도)
\n[6번 숫자 인식 결과] 11 / 15 성공 (73.3333% 정확도)

시도6. + 현재 증강(노이즈/쉬프트/밝기/컨트라스트 강화)와 six_bias 조정, 그리고 기존 Voting(최다 득표) 방식을 그대로 사용

void random_shift(float* img, int w, int h) {
    // int dx = (rand() % ) - 1; // -1, 0, 1 중 하나
    // int dy = (rand() % 3) - 1; // -1, 0, 1 중 하나
    int dx = (rand() % 5) - 2; // -2, -1, 0, 1, 2
    int dy = (rand() % 5) - 2;
    single precision: 14/15 (93.3%)
    half precision: 11/15 (73.3%)

single precision에서 매우 높은 인식률을 기록했으나,
half precision에서는 일부 이미지(특히 8, 5 등과 혼동)에서 오분류가 남아 있습니다.

시도7. softmax 평균 기반으로 변경
        // 누적 softmax 확률이 가장 큰 클래스를 예측값으로
        int best_id = 0;
        for (int i = 1; i < 10; i++) {
            if (prob_acc[best_id] < prob_acc[i]) best_id = i;
        }
    각 반복에서 나온 softmax 확률값(result[0]~result[9])을 모두 누적(accumulate)
    반복이 끝나면, 누적 확률이 가장 큰 클래스를 반환
    → 더 robust, bias나 증강에 덜 민감
--기존에 했던 식으로 six_bias= 12.0f로 했더니 기존 샘플 1,3,5 오분류 에러남.&& 7.0f도 오류, 6.0까지는 오류x
따라서 six_bias=6.0으로 재조정 & softmax 평균기반 변경

결과: 
\n[6번 숫자 인식 결과] 14 / 15 성공 (93.3333% 정확도)
\n[6번 숫자 인식 결과] 11 / 15 성공 (73.3333% 정확도)


------------실행시간 조정(nsys)----------------------
처음 코드 실행 시, 문제점
1. nsys로 본 현재 코드의 병목
    실행시간의 대부분이 cudaMalloc/cudaFree에 소모
        nsys 결과에서 cudaFree가 74.9%, cudaMalloc이 2.0% 등 할당/해제가 큰 비중을 차지합니다.
    cuDNN LRN(Local Response Normalization) 커널이 전체 GPU 커널 실행의 약 50%
    Host↔Device 또는 Device↔Device 메모리 복사도 상당한 시간 소모

2. 코드 구조의 실제 문제점
1) 비효율적 메모리 할당/해제

    classify_example 함수에서 convoluteForward, poolForward 등 각 레이어 함수가 내부적으로 매번 새로운 출력 버퍼를 할당(cudaMalloc)하고, 이전 입력 버퍼를 cudaFree 합니다.
    반복문(NUM_AUG)을 돌 때마다, 이미 할당된 포인터를 계속 덮어쓰기 때문에 불필요하게 메모리 할당/해제가 반복됩니다.

2) 불필요하거나 과도한 LRN 사용

    LRN(Local Response Normalization)은 오래된 네트워크에서 쓰이던 정규화 기법으로, 최근 네트워크에서는 거의 쓰이지 않고 오히려 속도만 느려집니다.
    nsys에서 LRN 관련 커널이 전체 커널 실행의 절반 가까이를 차지합니다.

3) 메모리 복사 패턴 비효율

    Host→Device 복사가 많고, 중간 결과도 Device간 복사가 빈번하게 발생합니다.

----
시도1. 할당/해제 반복 최소화: 반복문 밖에서 입력/출력 buffer를 할당, 내부에서는 포인터만 swap
이후 결과:
Failed to create '/home/user/GPU_yr/mnist_cudnn/mnistCUDNN/mnist_nsys_report.sqlite': 파일이 있습니다.
Use `--force-overwrite true` to overwrite existing files.
[2/8] [========================100%] nsys-report-b910.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /tmp/nsys-report-b910.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)     StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -------------  ----------------------
     52.0    1,356,089,216         57   23,791,038.9      806,279.0        1,590  162,110,405   40,176,303.3  poll                  
     44.3    1,154,574,117          3  384,858,039.0  500,089,613.0  154,391,026  500,093,478  199,590,288.0  pthread_cond_timedwait
      3.4       87,710,965      2,714       32,318.0       17,340.5        1,029    8,644,303      261,518.2  ioctl                 
      0.1        2,773,225         53       52,325.0       13,952.0        6,009      854,947      153,077.1  mmap64                
      0.0        1,248,478         34       36,719.9        6,973.5        1,372      210,487       67,492.1  mmap                  
      0.0        1,090,769        395        2,761.4        2,480.0        1,095       17,309        1,745.2  fopen                 
      0.0          824,124         18       45,784.7       43,604.5        6,943      207,853       44,583.6  sem_timedwait         
      0.0          776,029         25       31,041.2        1,425.0        1,001      512,968      107,029.9  read                  
      0.0          498,751        190        2,625.0        2,482.0        1,014       13,593          997.2  fread                 
      0.0          454,563         36       12,626.8        5,283.5        1,510      224,690       36,856.0  munmap                
      0.0          258,597         11       23,508.8       12,578.0        6,525       46,412       17,812.6  sem_wait              
      0.0          248,431        202        1,229.9        1,194.0        1,003        3,411          253.1  fclose                
      0.0          224,913          2      112,456.5      112,456.5       88,936      135,977       33,263.0  pthread_join          
      0.0          222,723         45        4,949.4        4,397.0        2,181       10,371        1,550.0  open64                
      0.0          171,007          1      171,007.0      171,007.0      171,007      171,007            0.0  pthread_mutex_lock    
      0.0          117,470          3       39,156.7       42,187.0       31,887       43,396        6,324.7  pthread_create        
      0.0           83,062         36        2,307.3        2,210.5        2,072        3,333          272.0  putc                  
      0.0           73,317          1       73,317.0       73,317.0       73,317       73,317            0.0  pthread_cond_wait     
      0.0           61,826         16        3,864.1        3,146.0        2,055        9,324        2,073.8  fopen64               
      0.0           36,698         23        1,595.6        1,483.0        1,058        2,993          419.5  write                 
      0.0           27,983          1       27,983.0       27,983.0       27,983       27,983            0.0  fgets                 
      0.0           18,494          7        2,642.0        2,235.0        1,440        3,763          864.3  fwrite                
      0.0           17,220          6        2,870.0        3,039.0        1,127        4,050        1,104.7  open                  
      0.0            9,462          2        4,731.0        4,731.0        2,930        6,532        2,547.0  socket                
      0.0            9,418          3        3,139.3        4,082.0        1,249        4,087        1,637.1  pipe2                 
      0.0            7,411          1        7,411.0        7,411.0        7,411        7,411            0.0  connect               
      0.0            3,903          1        3,903.0        3,903.0        3,903        3,903            0.0  pthread_kill          
      0.0            1,534          1        1,534.0        1,534.0        1,534        1,534            0.0  pthread_cond_broadcast
      0.0            1,454          1        1,454.0        1,454.0        1,454        1,454            0.0  pthread_cond_signal   
      0.0            1,372          1        1,372.0        1,372.0        1,372        1,372            0.0  bind                  

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)                Name              
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  -------------------------------
     74.5      806,463,976        448    1,800,142.8        6,849.0          420  600,477,901  29,240,258.6  cudaFree                       
     13.5      146,010,633          1  146,010,633.0  146,010,633.0  146,010,633  146,010,633           0.0  cudaDeviceReset                
      9.0       97,301,124      3,434       28,334.6        3,774.0          602   22,953,856     565,151.1  cudaLaunchKernel               
      1.2       12,870,391        441       29,184.6        7,633.0        3,026      127,451      27,122.3  cudaMalloc                     
      0.8        8,822,320        190       46,433.3       50,773.0          829       66,699      13,636.7  cudaDeviceSynchronize          
      0.5        5,876,999        736        7,985.1        7,498.5        3,956      169,832       7,463.5  cudaMemcpy                     
      0.1        1,380,822      1,130        1,222.0          743.5          259        7,208         772.6  cudaEventRecord                
      0.1        1,169,657          4      292,414.3      291,347.0        6,388      580,575     328,275.8  cudaHostAlloc                  
      0.1          818,510      1,040          787.0          830.0          251        2,292         300.2  cudaStreamWaitEvent            
      0.1          578,914          4      144,728.5      142,516.0        4,974      288,908     160,920.5  cudaFreeHost                   
      0.0          363,627         66        5,509.5        1,260.5        1,079       97,226      16,273.9  cudaStreamCreateWithFlags      
      0.0          338,580          6       56,430.0       56,765.0       54,507       58,135       1,603.9  cudaGetDeviceProperties        
      0.0          332,516         14       23,751.1       15,954.0        2,478       55,123      19,449.0  cudaEventSynchronize           
      0.0          282,409      2,282          123.8          103.0           58        4,478         109.7  cuGetProcAddress               
      0.0          127,590         90        1,417.7        1,400.0        1,289        2,426         130.4  cudaEventQuery                 
      0.0          116,229         66        1,761.0        1,406.5        1,120        6,051       1,147.6  cudaStreamDestroy              
      0.0           90,827        270          336.4          273.0          157        1,127         173.0  cudaStreamGetCaptureInfo_v10010
      0.0           71,697         14        5,121.2        2,760.0        2,170       16,706       4,421.4  cudaStreamBeginCapture_v10000  
      0.0           66,387          6       11,064.5       11,198.0          262       24,021       9,700.5  cudaMemsetAsync                
      0.0           28,218         14        2,015.6        1,518.5          868        9,366       2,154.2  cudaGraphDestroy_v10000        
      0.0           26,234         44          596.2          333.0          258        5,477         829.0  cudaEventCreateWithFlags       
      0.0           18,692         48          389.4          271.5          183          918         223.6  cudaEventDestroy               
      0.0           15,030         14        1,073.6        1,004.0          524        1,846         376.3  cudaStreamEndCapture_v10000    
      0.0            7,084          6        1,180.7        1,184.0          795        1,528         241.1  cuInit                         
      0.0            6,077          4        1,519.3        1,429.5        1,310        1,908         272.9  cudaEventCreate                
      0.0            4,017          2        2,008.5        2,008.5        1,333        2,684         955.3  cudaGetDriverEntryPoint_v11030 
      0.0              881          6          146.8          162.0           67          227          65.8  cuModuleGetLoadingMode         
      0.0              699          1          699.0          699.0          699          699           0.0  cuCtxSynchronize               

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     28.7        6,680,355         90  74,226.2  74,240.0    74,176    74,241         18.7  void cudnn::detail::lrn_fw_Nd_kernel<__half, float, (bool)1, (bool)0>(cudnnTensorStruct, const T1 *…
     19.6        4,569,743         90  50,774.9  50,784.0    50,752    50,816         16.1  void cudnn::detail::lrnForward_evenC<(int)5, float, float>(cudnn::detail::LrnForwardParams<T2, T3>) 
      3.2          755,810        160   4,723.8   4,720.5     4,640     4,928         43.3  void gemv2T_kernel_val<int, int, float2, float2, float2, float2, (int)128, (int)16, (int)2, (int)2,…
      3.1          712,193        160   4,451.2   4,416.0     3,520     5,408        902.9  void fft2d_r2c_16x16<__half>(float2 *, const T1 *, int, int, int, int, int, int, int, int)          
      3.0          691,813        160   4,323.8   5,024.0     3,488     5,248        802.3  void fft2d_r2c_16x16<float>(float2 *, const T1 *, int, int, int, int, int, int, int, int)           
      2.7          640,515         10  64,051.5  64,064.0    63,873    64,192        112.9  void explicit_convolve_sgemm<__half, int, (int)128, (int)6, (int)7, (int)3, (int)3, (int)5, (int)0,…
      2.7          618,625         82   7,544.2   7,552.0     7,456     7,680         35.4  void fft2d_c2r_32x32<__half, (bool)0, (bool)0, (unsigned int)0, (bool)0, (bool)0>(T1 *, const float…
      2.6          603,681         82   7,362.0   7,360.0     7,296     7,744         63.5  void fft2d_c2r_32x32<float, (bool)0, (bool)0, (unsigned int)0, (bool)0, (bool)0>(T1 *, const float2…
      2.3          545,665         20  27,283.3  26,752.0     4,320    50,592     23,506.3  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool)1, (bool)0, (bool)0, (bool)0, (bool)0,…
      2.3          541,952         81   6,690.8   6,688.0     6,592     6,752         26.4  void fft2d_r2c_32x32<__half, (bool)0, (unsigned int)5, (bool)0>(float2 *, const T1 *, int, int, int…
      2.3          541,120         81   6,680.5   6,688.0     6,592     6,848         50.4  void fft2d_r2c_32x32<float, (bool)0, (unsigned int)5, (bool)0>(float2 *, const T1 *, int, int, int,…
      2.1          498,240         90   5,536.0   5,568.0     4,960     5,792        146.8  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, float, float, float, float, (bool…
      1.9          443,842        180   2,465.8   2,432.0     2,304     2,592         85.5  void cudnn::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, float, (cudnnNanPropag…
      1.9          438,116         90   4,868.0   4,928.0     4,320     5,088        172.0  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, (b…
      1.8          416,257        180   2,312.5   2,304.0     2,176     2,433         94.3  void cudnn::pooling_fw_4d_kernel<__half, float, cudnn::maxpooling_func<float, __half, (cudnnNanProp…
      1.8          413,315        162   2,551.3   2,560.0     2,496     2,624         32.9  void gemmk1_kernel<int, float2, (int)256, (int)5, (bool)1, (bool)0, (bool)0, (bool)0, cublasGemvTen…
      1.6          373,377        180   2,074.3   2,080.0     2,048     2,272         26.0  void op_generic_tensor_kernel<(int)3, float, float, float, (int)256, (cudnnGenericOp_t)0, (cudnnNan…
      1.5          351,041        180   1,950.2   1,952.0     1,920     2,176         25.7  void op_generic_tensor_kernel<(int)3, __half, float, __half, (int)256, (cudnnGenericOp_t)0, (cudnnN…
      1.4          336,128         80   4,201.6   4,192.0     4,192     4,256         15.6  void fft2d_c2r_16x16<__half, (bool)0>(T1 *, float2 *, int, int, int, int, int, int, int, int, int, …
      1.4          335,041         80   4,188.0   4,192.0     4,160     4,256         14.7  void fft2d_c2r_16x16<float, (bool)0>(T1 *, float2 *, int, int, int, int, int, int, int, int, int, i…
      1.4          334,177         90   3,713.1   3,712.0     3,680     3,840         24.2  void gemv2T_kernel_val<int, int, float, float, float, float, (int)128, (int)16, (int)2, (int)2, (bo…
      1.3          313,600         82   3,824.4   3,808.0     3,808     4,128         40.6  void fft2d_r2c_32x32<__half, (bool)0, (unsigned int)0, (bool)0>(float2 *, const T1 *, int, int, int…
      1.2          285,024        161   1,770.3   1,536.0     1,344     2,176        382.7  void flip_filter<float, float>(T2 *, const T1 *, int, int, int, int)                                
      1.2          283,875        161   1,763.2   1,440.0     1,376     2,176        381.9  void flip_filter<__half, __half>(T2 *, const T1 *, int, int, int, int)                              
      1.1          264,705         82   3,228.1   3,200.0     3,168     3,520         60.4  void fft2d_r2c_32x32<float, (bool)0, (unsigned int)0, (bool)0>(float2 *, const T1 *, int, int, int,…
      0.9          220,865         10  22,086.5  22,080.0    22,048    22,112         20.2  void cudnn::engines_precompiled::im2col4d_kernel<__half, long>(cudnn::engines_precompiled::im2col4d…
      0.7          166,050         90   1,845.0   1,856.0     1,792     2,176         40.2  void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __hal…
      0.7          161,536         90   1,794.8   1,792.0     1,760     1,888         16.4  void reduce_1Block_kernel<float, (int)128, (int)7, cublasGemvTensorStridedBatched<float>, cublasGem…
      0.7          159,617         90   1,773.5   1,760.0     1,728     1,824         17.9  void op_generic_tensor_kernel<(int)1, float, float, float, (int)256, (cudnnGenericOp_t)8, (cudnnNan…
      0.6          148,000         90   1,644.4   1,632.0     1,600     1,696         17.8  void op_generic_tensor_kernel<(int)1, __half, float, __half, (int)256, (cudnnGenericOp_t)8, (cudnnN…
      0.6          140,480         90   1,560.9   1,568.0     1,504     1,600         15.0  void softmax_fw_small_kernel<float, float, (int)2, (int)1, (int)16, (int)1>(T1 *, const T1 *, int, …
      0.6          139,712         90   1,552.4   1,568.0     1,536     1,568         16.1  void softmax_fw_small_kernel<__half, float, (int)2, (int)1, (int)16, (int)1>(T1 *, const T1 *, int,…
      0.2           44,576         10   4,457.6   4,448.0     4,384     4,576         52.4  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool)1, (bool)0, (bool)0, (bool)0, (bool)0,…
      0.2           40,224          2  20,112.0  20,112.0     1,824    38,400     25,863.1  void cudnn::engines_precompiled::nchwToNhwcKernel<float, __half, float, (bool)0, (bool)1, (cudnnKer…
      0.1           17,760          4   4,440.0   4,368.0     4,160     4,864        344.0  void implicit_convolve_sgemm<__half, __half, (int)1024, (int)5, (int)5, (int)3, (int)3, (int)3, (in…
      0.1           12,704          3   4,234.7   4,224.0     3,968     4,512        272.2  void implicit_convolve_sgemm<float, float, (int)1024, (int)5, (int)5, (int)3, (int)3, (int)3, (int)…
      0.0            9,472          1   9,472.0   9,472.0     9,472     9,472          0.0  _5x_cudnn_ampere_scudnn_128x32_relu_interior_nn_v1                                                  
      0.0            8,320          2   4,160.0   4,160.0     4,160     4,160          0.0  void gemmk1_kernel<int, float, (int)256, (int)5, (bool)0, (bool)0, (bool)0, (bool)0, cublasGemvTens…
      0.0            6,816          1   6,816.0   6,816.0     6,816     6,816          0.0  void cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, __half>(cudnn::winograd_nonfused::…
      0.0            6,784          1   6,784.0   6,784.0     6,784     6,784          0.0  void cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::W…
      0.0            6,689          1   6,689.0   6,689.0     6,689     6,689          0.0  void fft2d_r2c_32x32<float, (bool)0, (unsigned int)5, (bool)1>(float2 *, const T1 *, int, int, int,…
      0.0            6,656          1   6,656.0   6,656.0     6,656     6,656          0.0  void fft2d_r2c_32x32<__half, (bool)0, (unsigned int)5, (bool)1>(float2 *, const T1 *, int, int, int…
      0.0            5,536          1   5,536.0   5,536.0     5,536     5,536          0.0  void cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, float>(cudnn::winograd_nonfused:…
      0.0            5,504          1   5,504.0   5,504.0     5,504     5,504          0.0  void cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, __half>(cudnn::winograd_nonfused…
      0.0            5,376          2   2,688.0   2,688.0     2,656     2,720         45.3  void gemmk1_kernel<int, float2, (int)256, (int)5, (bool)0, (bool)0, (bool)0, (bool)0, cublasGemvTen…
      0.0            2,560          2   1,280.0   1,280.0     1,216     1,344         90.5  void float2half_rn_kernel<float>(int, const T1 *, __half *)                                         
      0.0            2,432          1   2,432.0   2,432.0     2,432     2,432          0.0  void cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, __half>(cudnn::winograd_nonfused…
      0.0            2,400          1   2,400.0   2,400.0     2,400     2,400          0.0  void cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, float>(cudnn::winograd_nonfused:…
      0.0            1,569          1   1,569.0   1,569.0     1,569     1,569          0.0  void cask__5x_cudnn::computeOffsetsKernel<(bool)0, (bool)0>(cask__5x_cudnn::ComputeOffsetsParams)   

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation           
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
     42.4          444,289    360   1,234.1   1,248.0     1,184     1,408         24.3  [CUDA memcpy Device-to-Device]
     42.3          443,137    196   2,260.9     528.0       416   195,104     16,766.7  [CUDA memcpy Host-to-Device]  
     14.7          153,984    180     855.5     832.0       800     1,408         78.0  [CUDA memcpy Device-to-Host]  
      0.7            7,136      4   1,784.0   1,840.0     1,472     1,984        219.2  [CUDA memset]                 

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation           
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
      3.821    196     0.019     0.002     0.000     1.600        0.161  [CUDA memcpy Host-to-Device]  
      0.275    360     0.001     0.001     0.000     0.002        0.001  [CUDA memcpy Device-to-Device]
      0.052      4     0.013     0.013     0.013     0.013        0.000  [CUDA memset]                 
      0.005    180     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Device-to-Host]  

Generated:
	/tmp/nsys-report-2d57.nsys-rep
	/tmp/nsys-report-b910.sqlite

CUDA API 호출 타임라인

    cudaFree : 전체 CUDA API 시간의 74.5%
    cudaMalloc : 1.2%
    cudaMemcpy/DeviceSynchronize : 0.5%/0.8%
    cudaLaunchKernel : 9.0%
    cudaDeviceReset : 13.5%

해석:

    cudaFree, cudaMalloc은 대부분 초기화(가중치 등) 및 실행 후 해제에만 등장.
    반복적인 할당/해제가 아닌, 프로그램 전체에서 몇 번만 일어나고 있음(최적화 의도대로 잘 반영됨).
    이전에는 할당/해제 비중이 70~90%였던 것에 비해 대폭 줄었고,
    실제 커널 실행(cudaLaunchKernel)이 9%로 증가함.
    즉, 불필요한 malloc/free는 거의 사라짐!

시도2. LRN 레이어 제거
Generating '/tmp/nsys-report-99ad.qdstrm'
[1/8] [========================100%] mnist_nsys_report1.nsys-rep
[2/8] [========================100%] mnist_nsys_report1.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /home/user/GPU_yr/mnist_cudnn/mnistCUDNN/mnist_nsys_report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)     StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -------------  ----------------------
     51.8    1,374,948,237         63   21,824,575.2      234,238.0        4,645  163,339,585   39,822,525.6  poll                  
     44.0    1,168,088,533          3  389,362,844.3  500,056,410.0  167,947,797  500,084,326  191,751,056.3  pthread_cond_timedwait
      3.8      101,105,927      2,715       37,239.8       17,646.0        1,024   10,856,631      336,489.4  ioctl                 
      0.1        2,894,390         53       54,611.1       12,050.0        5,670      917,004      160,300.5  mmap64                
      0.0        1,226,457         34       36,072.3        6,875.0        2,396      209,721       65,989.3  mmap                  
      0.0        1,153,765        395        2,920.9        2,430.0        1,064       17,567        1,821.0  fopen                 
      0.0          840,340         18       46,685.6       45,797.0        5,384      211,634       46,043.3  sem_timedwait         
      0.0          792,609         28       28,307.5        1,566.5        1,003      536,543      104,544.7  read                  
      0.0          512,942        197        2,603.8        2,476.0        1,006        8,061          768.3  fread                 
      0.0          471,585         36       13,099.6        5,779.5        2,239      226,382       37,076.0  munmap                
      0.0          275,041        207        1,328.7        1,301.0        1,028        4,255          324.5  fclose                
      0.0          239,220          8       29,902.5       28,608.5       17,806       46,854        8,842.0  sem_wait              
      0.0          229,617          2      114,808.5      114,808.5       93,481      136,136       30,161.6  pthread_join          
      0.0          223,143         45        4,958.7        4,384.0        2,169       11,313        1,675.9  open64                
      0.0          204,972         23        8,911.8        1,604.0        1,188       26,920       11,225.5  write                 
      0.0          133,433          3       44,477.7       45,409.0       33,889       54,135       10,155.1  pthread_create        
      0.0           83,906         36        2,330.7        2,310.5        2,054        3,417          281.5  putc                  
      0.0           73,357          1       73,357.0       73,357.0       73,357       73,357            0.0  pthread_cond_wait     
      0.0           61,032         16        3,814.5        3,062.5        2,036       10,529        2,532.7  fopen64               
      0.0           27,669          1       27,669.0       27,669.0       27,669       27,669            0.0  fgets                 
      0.0           18,101          6        3,016.8        3,370.0        1,018        4,278        1,252.5  open                  
      0.0           16,949          7        2,421.3        2,176.0        1,049        3,995        1,064.2  fwrite                
      0.0            9,765          3        3,255.0        3,888.0        1,407        4,470        1,626.7  pipe2                 
      0.0            9,646          2        4,823.0        4,823.0        3,571        6,075        1,770.6  socket                
      0.0            7,982          1        7,982.0        7,982.0        7,982        7,982            0.0  connect               
      0.0            3,788          1        3,788.0        3,788.0        3,788        3,788            0.0  pthread_kill          
      0.0            2,049          1        2,049.0        2,049.0        2,049        2,049            0.0  pthread_cond_signal   
      0.0            1,572          1        1,572.0        1,572.0        1,572        1,572            0.0  pthread_cond_broadcast
      0.0            1,401          1        1,401.0        1,401.0        1,401        1,401            0.0  bind                  

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)                Name              
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  -------------------------------
     75.0      824,993,640        448    1,841,503.7        6,917.0          516  608,505,757  29,657,909.4  cudaFree                       
     13.9      152,549,154          1  152,549,154.0  152,549,154.0  152,549,154  152,549,154           0.0  cudaDeviceReset                
      8.8       97,066,154      3,254       29,829.8        3,858.0          703   23,102,125     584,169.7  cudaLaunchKernel               
      1.2       13,146,243        441       29,810.1        7,761.0        2,800      112,436      27,218.4  cudaMalloc                     
      0.5        5,811,125        736        7,895.6        7,459.0        4,222      108,481       5,984.1  cudaMemcpy                     
      0.1        1,400,295      1,130        1,239.2          794.0          239        7,418         767.9  cudaEventRecord                
      0.1        1,184,653          4      296,163.3      289,183.5        7,121      599,165     333,151.6  cudaHostAlloc                  
      0.1          826,711      1,040          794.9          836.5          254        2,158         312.2  cudaStreamWaitEvent            
      0.1          619,737         14       44,266.9       42,183.0       27,660       67,222      13,916.7  cudaEventSynchronize           
      0.1          614,775        190        3,235.7        3,198.0          778        7,285         607.7  cudaDeviceSynchronize          
      0.1          571,822          4      142,955.5      143,234.5        5,132      280,221     156,722.5  cudaFreeHost                   
      0.0          370,757          6       61,792.8       58,982.5       55,919       78,945       8,649.2  cudaGetDeviceProperties        
      0.0          356,496         66        5,401.5        1,249.5        1,062       91,399      14,926.7  cudaStreamCreateWithFlags      
      0.0          282,411      2,282          123.8          104.0           59        1,056          67.0  cuGetProcAddress               
      0.0          137,651         66        2,085.6        1,588.5        1,149        9,710       1,517.8  cudaStreamDestroy              
      0.0          134,280         90        1,492.0        1,482.0        1,337        2,177         112.7  cudaEventQuery                 
      0.0           88,443        270          327.6          283.0          152        1,015         150.2  cudaStreamGetCaptureInfo_v10010
      0.0           75,655          6       12,609.2       11,361.5          273       27,899      11,663.4  cudaMemsetAsync                
      0.0           63,577         14        4,541.2        2,884.0        2,500       13,944       3,152.8  cudaStreamBeginCapture_v10000  
      0.0           29,065         14        2,076.1        1,607.5          962        9,598       2,216.6  cudaGraphDestroy_v10000        
      0.0           28,026         44          637.0          338.5          262        5,481         871.9  cudaEventCreateWithFlags       
      0.0           23,817         48          496.2          290.0          179        3,985         581.5  cudaEventDestroy               
      0.0           15,680         14        1,120.0        1,040.0          657        1,976         414.3  cudaStreamEndCapture_v10000    
      0.0            7,865          6        1,310.8        1,427.0          820        1,620         304.9  cuInit                         
      0.0            7,020          4        1,755.0        1,693.5        1,566        2,067         227.7  cudaEventCreate                
      0.0            3,912          2        1,956.0        1,956.0        1,387        2,525         804.7  cudaGetDriverEntryPoint_v11030 
      0.0              823          6          137.2           96.5           87          301          83.8  cuModuleGetLoadingMode         
      0.0              711          1          711.0          711.0          711          711           0.0  cuCtxSynchronize               

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
      6.4          757,543        160   4,734.6   4,736.0     4,640     4,896         39.4  void gemv2T_kernel_val<int, int, float2, float2, float2, float2, (int)128, (int)16, (int)2, (int)2,…
      6.1          717,411        160   4,483.8   4,704.0     3,552     5,472        908.4  void fft2d_r2c_16x16<__half>(float2 *, const T1 *, int, int, int, int, int, int, int, int)          
      5.9          692,516        160   4,328.2   4,320.0     3,456     5,280        812.7  void fft2d_r2c_16x16<float>(float2 *, const T1 *, int, int, int, int, int, int, int, int)           
      5.4          638,918         10  63,891.8  63,809.0    63,745    64,385        195.2  void explicit_convolve_sgemm<__half, int, (int)128, (int)6, (int)7, (int)3, (int)3, (int)5, (int)0,…
      5.3          622,279         82   7,588.8   7,584.0     7,520     7,680         27.6  void fft2d_c2r_32x32<__half, (bool)0, (bool)0, (unsigned int)0, (bool)0, (bool)0>(T1 *, const float…
      5.1          601,795         82   7,339.0   7,328.0     7,264     7,712         64.6  void fft2d_c2r_32x32<float, (bool)0, (bool)0, (unsigned int)0, (bool)0, (bool)0>(T1 *, const float2…
      4.6          541,381         81   6,683.7   6,688.0     6,592     6,816         47.4  void fft2d_r2c_32x32<float, (bool)0, (unsigned int)5, (bool)0>(float2 *, const T1 *, int, int, int,…
      4.6          540,256         81   6,669.8   6,688.0     6,592     6,720         35.8  void fft2d_r2c_32x32<__half, (bool)0, (unsigned int)5, (bool)0>(float2 *, const T1 *, int, int, int…
      4.2          497,733         90   5,530.4   5,536.0     4,992     5,792        142.8  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, float, float, float, float, (bool…
      3.7          438,405         90   4,871.2   4,928.0     4,160     5,057        215.6  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, (b…
      3.7          433,924        180   2,410.7   2,352.5     2,304     2,624         84.8  void cudnn::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, float, (cudnnNanPropag…
      3.5          415,300        162   2,563.6   2,560.0     2,496     2,656         29.0  void gemmk1_kernel<int, float2, (int)256, (int)5, (bool)1, (bool)0, (bool)0, (bool)0, cublasGemvTen…
      3.5          408,199        180   2,267.8   2,304.0     2,176     2,433         79.4  void cudnn::pooling_fw_4d_kernel<__half, float, cudnn::maxpooling_func<float, __half, (cudnnNanProp…
      3.2          372,640        180   2,070.2   2,080.0     2,016     2,240         25.1  void op_generic_tensor_kernel<(int)3, float, float, float, (int)256, (cudnnGenericOp_t)0, (cudnnNan…
      3.0          352,224        180   1,956.8   1,952.0     1,920     2,176         24.7  void op_generic_tensor_kernel<(int)3, __half, float, __half, (int)256, (cudnnGenericOp_t)0, (cudnnN…
      2.9          336,929         80   4,211.6   4,224.0     4,192     4,256         17.2  void fft2d_c2r_16x16<__half, (bool)0>(T1 *, float2 *, int, int, int, int, int, int, int, int, int, …
      2.8          334,595         80   4,182.4   4,192.0     4,160     4,256         17.9  void fft2d_c2r_16x16<float, (bool)0>(T1 *, float2 *, int, int, int, int, int, int, int, int, int, i…
      2.8          331,363         90   3,681.8   3,680.0     3,648     3,712         16.8  void gemv2T_kernel_val<int, int, float, float, float, float, (int)128, (int)16, (int)2, (int)2, (bo…
      2.7          318,305         82   3,881.8   3,872.0     3,840     4,096         35.1  void fft2d_r2c_32x32<__half, (bool)0, (unsigned int)0, (bool)0>(float2 *, const T1 *, int, int, int…
      2.6          302,146         22  13,733.9   4,480.0     3,616    33,280     11,210.0  void implicit_convolve_sgemm<float, float, (int)1024, (int)5, (int)5, (int)3, (int)3, (int)3, (int)…
      2.4          285,441        161   1,772.9   1,536.0     1,344     2,176        384.4  void flip_filter<float, float>(T2 *, const T1 *, int, int, int, int)                                
      2.4          284,259        161   1,765.6   1,440.0     1,344     2,176        383.2  void flip_filter<__half, __half>(T2 *, const T1 *, int, int, int, int)                              
      2.3          268,931         82   3,279.6   3,233.0     3,200     3,552         66.5  void fft2d_r2c_32x32<float, (bool)0, (unsigned int)0, (bool)0>(float2 *, const T1 *, int, int, int,…
      1.9          220,865         10  22,086.5  22,080.0    22,048    22,112         20.2  void cudnn::engines_precompiled::im2col4d_kernel<__half, long>(cudnn::engines_precompiled::im2col4d…
      1.4          164,385         90   1,826.5   1,824.0     1,792     1,920         19.9  void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __hal…
      1.4          159,363         90   1,770.7   1,760.0     1,728     1,824         17.3  void op_generic_tensor_kernel<(int)1, float, float, float, (int)256, (cudnnGenericOp_t)8, (cudnnNan…
      1.3          158,403         90   1,760.0   1,760.0     1,696     1,888         20.4  void reduce_1Block_kernel<float, (int)128, (int)7, cublasGemvTensorStridedBatched<float>, cublasGem…
      1.3          148,162         90   1,646.2   1,632.0     1,632     1,696         17.4  void op_generic_tensor_kernel<(int)1, __half, float, __half, (int)256, (cudnnGenericOp_t)8, (cudnnN…
      1.2          140,416         90   1,560.2   1,568.0     1,504     1,600         16.1  void softmax_fw_small_kernel<float, float, (int)2, (int)1, (int)16, (int)1>(T1 *, const T1 *, int, …
      1.2          140,385         90   1,559.8   1,568.0     1,536     1,600         14.8  void softmax_fw_small_kernel<__half, float, (int)2, (int)1, (int)16, (int)1>(T1 *, const T1 *, int,…
      0.4           44,768         10   4,476.8   4,448.0     4,416     4,640         63.0  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool)1, (bool)0, (bool)0, (bool)0, (bool)0,…
      0.3           40,544          2  20,272.0  20,272.0     1,856    38,688     26,044.2  void cudnn::engines_precompiled::nchwToNhwcKernel<float, __half, float, (bool)0, (bool)1, (cudnnKer…
      0.1           17,697          4   4,424.3   4,336.5     4,160     4,864        336.8  void implicit_convolve_sgemm<__half, __half, (int)1024, (int)5, (int)5, (int)3, (int)3, (int)3, (in…
      0.1            9,440          1   9,440.0   9,440.0     9,440     9,440          0.0  _5x_cudnn_ampere_scudnn_128x32_relu_interior_nn_v1                                                  
      0.1            8,352          2   4,176.0   4,176.0     4,160     4,192         22.6  void gemmk1_kernel<int, float, (int)256, (int)5, (bool)0, (bool)0, (bool)0, (bool)0, cublasGemvTens…
      0.1            6,816          1   6,816.0   6,816.0     6,816     6,816          0.0  void cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, __half>(cudnn::winograd_nonfused::…
      0.1            6,784          1   6,784.0   6,784.0     6,784     6,784          0.0  void cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::W…
      0.1            6,720          1   6,720.0   6,720.0     6,720     6,720          0.0  void fft2d_r2c_32x32<__half, (bool)0, (unsigned int)5, (bool)1>(float2 *, const T1 *, int, int, int…
      0.1            6,688          1   6,688.0   6,688.0     6,688     6,688          0.0  void fft2d_r2c_32x32<float, (bool)0, (unsigned int)5, (bool)1>(float2 *, const T1 *, int, int, int,…
      0.0            5,568          1   5,568.0   5,568.0     5,568     5,568          0.0  void cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, __half>(cudnn::winograd_nonfused…
      0.0            5,536          1   5,536.0   5,536.0     5,536     5,536          0.0  void cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, float>(cudnn::winograd_nonfused:…
      0.0            5,472          2   2,736.0   2,736.0     2,720     2,752         22.6  void gemmk1_kernel<int, float2, (int)256, (int)5, (bool)0, (bool)0, (bool)0, (bool)0, cublasGemvTen…
      0.0            4,384          1   4,384.0   4,384.0     4,384     4,384          0.0  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool)1, (bool)0, (bool)0, (bool)0, (bool)0,…
      0.0            2,528          2   1,264.0   1,264.0     1,216     1,312         67.9  void float2half_rn_kernel<float>(int, const T1 *, __half *)                                         
      0.0            2,432          1   2,432.0   2,432.0     2,432     2,432          0.0  void cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, __half>(cudnn::winograd_nonfused…
      0.0            2,432          1   2,432.0   2,432.0     2,432     2,432          0.0  void cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, float>(cudnn::winograd_nonfused:…
      0.0            1,568          1   1,568.0   1,568.0     1,568     1,568          0.0  void cask__5x_cudnn::computeOffsetsKernel<(bool)0, (bool)0>(cask__5x_cudnn::ComputeOffsetsParams)   

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation           
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
     44.0          444,098    360   1,233.6   1,248.0     1,184     1,408         22.3  [CUDA memcpy Device-to-Device]
     39.3          396,547    196   2,023.2     528.5       416   142,305     14,105.7  [CUDA memcpy Host-to-Device]  
     16.0          161,856    180     899.2     864.0       832     1,440         86.4  [CUDA memcpy Device-to-Host]  
      0.7            7,073      4   1,768.3   1,824.0     1,472     1,953        206.7  [CUDA memset]                 

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation           
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
      3.821    196     0.019     0.002     0.000     1.600        0.161  [CUDA memcpy Host-to-Device]  
      0.275    360     0.001     0.001     0.000     0.002        0.001  [CUDA memcpy Device-to-Device]
      0.052      4     0.013     0.013     0.013     0.013        0.000  [CUDA memset]                 
      0.005    180     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Device-to-Host]  

Generated:
	/home/user/GPU_yr/mnist_cudnn/mnistCUDNN/mnist_nsys_report1.nsys-rep
	/home/user/GPU_yr/mnist_cudnn/mnistCUDNN/mnist_nsys_report1.sqlite

| Name                             | Time(%) | Total(ns) | 설명                 |
| -------------------------------- | ------- | --------- | ------------------ |
| **gemv2T\_kernel 등 FC/행렬곱**      | 6.4     | 751,780   | Fully Connected 연산 |
| **fft2d\_r2c\_16x16 등 FFT/Conv** | \~6%    | \~700,000 | Conv, FFT 등        |
| pooling, softmax, etc.           | 3\~6%   | \~400,000 | 풀링, 소프트맥스 등        |
| **LRN 계열 없음!!**                  | **0%**  | **0**     | (제거됨)              |
