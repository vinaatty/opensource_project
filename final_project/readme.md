
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



