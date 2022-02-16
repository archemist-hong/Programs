<br>

#  복합 방법론을 적용한 불균형 자료의 이진 분류 프로그램

# (Binary classification program for imbalanced data using combination method)

<br>

<br>

## 프로그램 개요

<br>

<br>

 여러개의 ‘\*.csv’ 파일로 주어진 데이터 셋에 대해, GEV-활성함수, SMOTE(Synthetic Minority Over-sampling Technique), Focal Loss, Thresholding(Threshold-moving)과 같은 방법론을 선택 적용한 인공 신경망에 기반하여 클래스 불균형 자료를 이진 분류한다. 이후, 학습된 모델을 ‘\*.pt’ 파일로 저장하며, 6가지 평가지표에 따라 학습된 모델에 대한 평가를 진행하고 평가 결과 및 학습과정을 그림 또는 ‘*.csv’파일로 저장한다.

<br>

## 주요 기능

<br>

<br>

1. 인공 신경망 기반 이진 분류 예측 모형 제공

- ‘Data Source’ 파일 경로에 주어진 여러개의 ‘\*.csv’ 파일로부터 인공 신경망 기반의 이진 분류 예측 모형을 학습시키고, 학습된 모형을 ‘\*.pt’ 파일로 저장한다. 인공 신경망 학습에 필요한 10가지 hyper parameter (activation function, learning rate 등)는 프로그램의 옵션을 통해 사용자가 선택할 수 있다.

​    

2. 모형 진단을 위한 학습 과정 이미지 파일 제공

   - 모형이 잘 학습되었는지 진단하기 위해서, 학습 과정을 이미지 파일로 제공한다. 제공되는 이미지는 

   - 1. 모수 추정 과정(GEV-활성함수를 사용한 경우), 

     2. 학습 과정의 Train / Test Loss 변화 추이와 Early Stopping Check point로 

        1개의 모델당 2개의 학습 과정 이미지가 제공된다.

​    

3. 모형 진단을 위한 평가 이미지, 6종의 평가지표 값 제공

- 학습이 완료된 모델에 대하여, 학습이 잘 진행되었는지 검증할 수 있는 평가 이미지(ROC curve) 및 6종의 평가지표(F1-score, Geometric Mean, Area Under ROC Curve, Balanced Accuracy, Brier Inaccuracy, Cohen’s Kappa)가 제공되며, 최종 모델의 Test Loss와 GEV-활성함수를 사용한 경우, 3개의 모수 mu, sigma, xi의 최종 추정값이 ‘\*.csv’ 파일로 제공된다.

<br>

## 사용 방법

<br>

<br>

 python 및 관련 패키지들이 설치된 PC에서 명령 프롬프트를 이용하여 ‘main.py’ 파일이 있는 경로에 접근한다. 

이후 “python main.py F:/imb_class -s F:/imb_class/data” 등의 명령어를 통해 프로그램 옵션을 조정하여 실행한다. 

프로그램 옵션에 대한 자세한 도움말은 “python main.py --help” 명령어를 통해 확인할 수 있다.

<br>

### Arguments

<br>

1. working_directory : 필수. Ex) F:/imb_class

2. source (-s): 필수. Ex)  F:/imb_class/data

3. oversampling (-os) : Float 형 실수. Ex) 0.1, Default : 1/20

   - Mi : Number of Samples in Minority Class, Ma : Number of Samples in Majority Class After Over sampling

   $$
   {Mi} \over {Ma}
   $$

4. seed (-r) : 정수들의 리스트. Ex) 20210905 20210906 20210907, Default : 20210905 ~ 20210908

5. testsize (-t) : Float 형 실수. Ex) 0.3, Default : 0.3

6. validsize (-v) : Float 형 실수. Ex) 0.2, Default : 0.2

7. learningrate (-l) : Float 형 실수. Ex) 0.001, Default : 0.001

8. batchsize (-b) : Int형 정수. Ex) 32, Default : 32

9. epochs (-e) : Int형 정수. Ex) 2000, Default : 2000

10. patience (-p) : Int형 정수. Ex) 20, Default : 20

11. showtraining (-show) : bool. Ex) True, Default : False

12. lossfunction (-loss) : String. Ex) BCE, Default : Focal, choices = ['Focal', 'BCE']

13. activation (-a) : String. Ex) Sigmoid, Default : GEV, choices = ['Sigmoid', 'GEV']

<br>

## Examples

<br>

<br>

### 실행 예제

<br>

``` bash
$ python imbalance_classification.py F:/imb_class -s F:/imb_class/data # 기본적인 실행
```

```bash
$ python imbalance_classification.py F:/imb_class -s F:/imb_class/data -r 1 2 -t 0.2 -v 0.2 -e 200 -p 5 -loss Focal -a Sigmoid # 옵션을 준 경우
```

<br>

### 도움말 예제

<br>

```bash
$ python imbalance_classification.py --help
```

<br>

## Dependencies

<br>

<br>

<img src = "https://img.shields.io/badge/python-3.8-blue" /> 

<img src = "https://img.shields.io/badge/torch-1.9.0-red" /> <img src = "https://img.shields.io/badge/tensorflow-2.4.1-red" /><img src = "https://img.shields.io/badge/imblearn-0.9.0-black" />

<img src = "https://img.shields.io/badge/sklearn-0.24.2-green" /> <img src = "https://img.shields.io/badge/scipy-1.6.2-green" /> <img src = "https://img.shields.io/badge/pandas-1.3.1-green" /> <img src = "https://img.shields.io/badge/numpy-1.19.2-green" /> <img src = "https://img.shields.io/badge/matplotlib-3.4.2-green" />

이 외에 os, random 을 사용합니다.

pytorch 및 tensorflow는 cpu version을 사용합니다.

<br>

## 관련 연구

<br>

<br>

국가과제 고유 번호 : 1345334971

연구 과제명 : 산업 빅데이터의 융복합 교육 연구단

연구 기간 : 2021-03-01 ~ 2022-02-28

주관 기관 : 전남대학교

<br>

# 공개 자료

<br>

### 학술 회의

<br>

| 연번 | 성명                                | 학술회의명                                                   | 개최국가 | 개최일     | 발표논문명                                                   |
| ---- | ----------------------------------- | ------------------------------------------------------------ | -------- | ---------- | ------------------------------------------------------------ |
| 1    | 홍주영, 신용관, 박혜빈, 박정수      | 한국데이터마이닝학회 2021 추계 학술대회                      | 대한민국 | 2021.11.25 | [클래스 불균형 데이터 분류를 위한 GEV 활성 함수에 관한 연구](./공개자료/2021KDMS-PPT.pdf) |
| 2    | H.Park, J.Hong, Y.Shin and J-S.Park | IES 2022 (INNOVATION AND SOCIETY 5.0: STTATISTICAL AND ECONOMIC METHODOLOGIES FOR QUALITY ASSESSMENT) | Italy    | 2022.01.27 | [A study on the GEV activation function for classification of class imbalance data](./공개자료/IES2022발표자료.pdf) |

<br>

## 학위 논문

<br>

홍주영. (2022). 클래스 불균형 데이터 분류를 위한 GEV 활성 함수에 관한 연구. 석사학위논문, 전남대학교, 광주.

<br>

<br>

## License

[복합 방법론을 적용한 불균형 자료의 이진 분류 프로그램](./공개자료/프로그램등록 정보.PNG)
