# KYOWAN X DACON AI OCR
## 1. 개요
https://dacon.io/competitions/official/236042/overview/description
  - 주제 : 손글씨 인식 AI 모델 개발
  - Task : Text Recognition, OCR
  - 기간 : 2022.12.26 ~ 2023.01.16
  - 결과 : 26등 / 430
<!--  Other options to write Readme
  - [Deployment](#deployment)
  - [Used or Referenced Projects](Used-or-Referenced-Projects)
-->
## 2. 데이터셋 설명
<!--Wirte one paragraph of project description --> 
- train(폴더) : 폰트 손글씨 학습 데이터, TRAIN_00000.png ~ TRAIN_76887.png


- test(폴더) :  폰트 손글씨 평가 데이터, TEST_00000.png ~ TEST_74120.png


- train.csv
  - id : 샘플 고유 id
  - img_path : 샘플 이미지 파일 경로
  - label : 샘플 이미지에 해당하는 Text


- test.csv
  - id : 샘플 고유 id
  - img_path : 샘플 이미지 파일 경로



## 3. 수행방법
<!-- Write Overview about this project -->
- 본 과제는 블랙박스 영상으로부터 자동차의 충돌 상황을 분석하는 AI 모델을 개발하는 것
- 본 데이터의 LABEL의 위의 이미지처럼 분할이 가능함
- label을 crash, weather, timing으로 분할하여 multi_label_classification 문제로 변환
- 모델로는 slowfast_r101, MVITv2_B_32x3, r3d_18을 사용해본 결과, r3d_18이 성능이 가장 좋았음
- 최종적으로 carsh는 r3d_18 모델 사용, weather, timing은 영상에서 각 5개의 이미지를 random으로 추출하고 convnext_large 모델 사용
- 각 label별로 병렬적으로 임베딩 추출하여 concat
- 최종적으로 F1-score 0.57464 달성

## 4. 한계점
<!-- Write Overview about this project -->
- 데이터 특성상 클래스 불균형도 심하고 잘못 labeling 된 데이터도 포함되어 있어서 쉽지 않은 대회였음

## Team member
장종환 (개인 참가)

