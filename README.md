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
<img width="275" alt="image" src="https://github.com/jang3463/kyowon_ai_ocr/assets/70848146/a797e391-b644-40af-8ae9-4981f5a5a148">  

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
- 본 과제는 손글씨 폰트 이미지를 바탕으로 Text Recognition을 수행하는 인식 AI 모델을 개발하는 것
- 네이버 클로바 deep-text-recognition 모델에 VITSTR 모델 추가 적용
- CUTMIX 기법을 사용해서 data augmentation 적용
- 최종적으로 accuracy 0.87485 달성

## 4. 한계점
<!-- Write Overview about this project -->
- 화질이 낮거나 노이즈가 있는 이미지가 섞여있었음
- 이 문제를 해결하기위해서 denoising이나 super resolution을 적용후 text-recognition을 했다면 더 좋은 성능을 얻을 수 있을거라고 생각함

## Team member
장종환 (개인 참가)

