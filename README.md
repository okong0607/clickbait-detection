# 뉴스 헤드라인 기반 낚시성(Clickbait) 기사 탐지

## 프로젝트 개요
자극적인 제목으로 독자를 유인하는 낚시성 기사를 자동으로 탐지하는 이진 분류 모델입니다.
뉴스 헤드라인만으로 낚시성 여부를 판별하여 미디어 플랫폼의 신뢰도를 높이는 것을 목표로 합니다.

## 데이터셋
- **출처**: [AI Hub - 낚시성 기사 탐지 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71338)
- **규모**: 660,085건 (Train 586,741 / Val 73,344)
- **클래스**: 정상(0) 333,054건 / 낚시성(1) 327,031건

## 모델 구성
| 모델 | 설명 |
|------|------|
| Baseline | TF-IDF (char n-gram) + Logistic Regression |
| Advanced | KcBERT (`beomi/kcbert-base`) 파인튜닝 |

## 실험 결과
| 모델 | Accuracy | F1-score |
|------|----------|----------|
| TF-IDF + LR (Baseline) | 0.6008 | 0.5960 |
| KcBERT (Advanced) | 0.6176 | 0.6057 |

## 실행 방법

### 1. 환경 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 전처리
AI Hub에서 데이터 다운로드 후:
```bash
python preprocess.py
```

### 3. 학습 및 평가 (로컬)
```bash
python train.py
```

### 4. Colab에서 실행
`clickbait_detection.ipynb`를 Google Colab에서 열고 T4 GPU 설정 후 실행

## 파일 구조
```
clickbait_detection/
├── preprocess.py            # 데이터 전처리 (JSON → CSV)
├── train.py                 # 로컬 학습 스크립트
├── clickbait_detection.ipynb  # Colab 노트북
├── requirements.txt         # 의존성 패키지
└── results/                 # 실험 결과 이미지 및 CSV
```

## 개발 환경
- Python 3.10
- PyTorch 2.2.0
- Transformers 4.40.0
- scikit-learn 1.4.0

## AI 도구 사용 고지
본 프로젝트에서 Claude (Anthropic)를 활용하여 코드 구조 설계 및 디버깅을 보조하였습니다.
