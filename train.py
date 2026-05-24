"""
낚시성 기사 탐지 - 전체 학습 파이프라인
Baseline: TF-IDF + Logistic Regression
Advanced: KcBERT 파인튜닝
결과: results/ 폴더에 저장
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

# ── 경로 설정 ────────────────────────────────────────────────────
BASE   = Path(r"C:\Users\jiwoo-choi\Desktop\clickbait_detection")
DATA   = BASE / "clickbait_data.csv"
RESULT = BASE / "results"
RESULT.mkdir(exist_ok=True)

# ── 한글 폰트 ────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── 1. 데이터 로드 ───────────────────────────────────────────────
print("\n[1] 데이터 로드 중...")
df = pd.read_csv(DATA)
print(f"  전체: {len(df):,}건")

# 10만 건 샘플링 (속도/메모리)
df = df.sample(n=100_000, random_state=42).reset_index(drop=True)
X, y = df["title"], df["clickbait_class"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val,   X_test,  y_val,   y_test  = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ── 2. EDA 저장 ──────────────────────────────────────────────────
print("\n[2] EDA 저장 중...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["clickbait_class"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color=["steelblue", "tomato"], edgecolor="white"
)
axes[0].set_title("클래스 분포")
axes[0].set_xticklabels(["정상(0)", "낚시성(1)"], rotation=0)
axes[0].set_ylabel("개수")

df["title_len"] = df["title"].str.len()
for label, color in [(0, "steelblue"), (1, "tomato")]:
    df[df["clickbait_class"] == label]["title_len"].plot(
        kind="kde", ax=axes[1], color=color, label=["정상", "낚시성"][label]
    )
axes[1].set_title("라벨별 제목 길이 분포")
axes[1].set_xlabel("글자 수")
axes[1].legend()
plt.tight_layout()
plt.savefig(RESULT / "eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → results/eda.png 저장")

# ── 3. Baseline: TF-IDF + Logistic Regression ───────────────────
print("\n[3] Baseline 학습 중...")
tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), analyzer="char_wb")
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)

lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_train_tfidf, y_train)

bl_pred = lr.predict(X_test_tfidf)
bl_acc  = accuracy_score(y_test, bl_pred)
bl_f1   = f1_score(y_test, bl_pred)
print(f"  Baseline Test  Accuracy: {bl_acc:.4f} | F1: {bl_f1:.4f}")

cm = confusion_matrix(y_test, bl_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["정상", "낚시성"], yticklabels=["정상", "낚시성"])
plt.title("Baseline Confusion Matrix")
plt.ylabel("실제"); plt.xlabel("예측")
plt.tight_layout()
plt.savefig(RESULT / "baseline_cm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → results/baseline_cm.png 저장")

# ── 4. Advanced: KcBERT 파인튜닝 ────────────────────────────────
print("\n[4] KcBERT 파인튜닝 중...")
MODEL_NAME = "beomi/kcbert-base"
MAX_LEN    = 64
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ClickbaitDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx], max_length=MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_loader = DataLoader(ClickbaitDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(ClickbaitDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(ClickbaitDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
    num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

def evaluate(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            )
            preds.extend(out.logits.argmax(-1).cpu().numpy())
            trues.extend(batch["label"].numpy())
    return accuracy_score(trues, preds), f1_score(trues, preds), trues, preds

history = {"loss": [], "val_acc": [], "val_f1": []}
best_f1 = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        out  = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            labels=batch["label"].to(DEVICE)
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total_loss += out.loss.item()

        if (step + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} Step {step+1}/{len(train_loader)} loss={out.loss.item():.4f}")

    val_acc, val_f1, _, _ = evaluate(val_loader)
    avg_loss = total_loss / len(train_loader)
    history["loss"].append(avg_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    print(f"  [Epoch {epoch+1}] loss={avg_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        model.save_pretrained(BASE / "best_model")
        tokenizer.save_pretrained(BASE / "best_model")
        print("  → 모델 저장")

# 학습 곡선
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, EPOCHS+1), history["loss"], marker="o")
axes[0].set_title("Train Loss"); axes[0].set_xlabel("Epoch")
axes[1].plot(range(1, EPOCHS+1), history["val_acc"], marker="o", label="Accuracy")
axes[1].plot(range(1, EPOCHS+1), history["val_f1"],  marker="s", label="F1")
axes[1].set_title("Validation 성능"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.tight_layout()
plt.savefig(RESULT / "training_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5. 최종 평가 ─────────────────────────────────────────────────
print("\n[5] 최종 평가 중...")
bert_acc, bert_f1, y_true, y_pred = evaluate(test_loader)
print(f"  KcBERT  Test  Accuracy: {bert_acc:.4f} | F1: {bert_f1:.4f}")

summary = pd.DataFrame({
    "모델":      ["TF-IDF + LR (Baseline)", "KcBERT (Advanced)"],
    "Accuracy": [round(bl_acc,  4), round(bert_acc, 4)],
    "F1-score": [round(bl_f1,   4), round(bert_f1,  4)],
})
summary.to_csv(RESULT / "model_comparison.csv", index=False, encoding="utf-8-sig")
print("\n=== 모델 비교 ===")
print(summary.to_string(index=False))
print("\n", classification_report(y_true, y_pred, target_names=["정상", "낚시성"]))

cm2 = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["정상", "낚시성"], yticklabels=["정상", "낚시성"])
plt.title("KcBERT Confusion Matrix")
plt.ylabel("실제"); plt.xlabel("예측")
plt.tight_layout()
plt.savefig(RESULT / "bert_cm.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6. 오분류 분석 ───────────────────────────────────────────────
print("\n[6] 오분류 분석 저장 중...")
test_df = X_test.reset_index(drop=True).to_frame()
test_df["true"]    = y_true
test_df["pred"]    = y_pred
fp = test_df[(test_df["true"] == 0) & (test_df["pred"] == 1)].head(20)
fn = test_df[(test_df["true"] == 1) & (test_df["pred"] == 0)].head(20)
error_df = pd.concat([
    fp.assign(error_type="FP (정상→낚시성 오탐)"),
    fn.assign(error_type="FN (낚시성→정상 미탐)"),
])
error_df.to_csv(RESULT / "error_analysis.csv", index=False, encoding="utf-8-sig")
print("  → results/error_analysis.csv 저장")

print("\n✓ 완료! results/ 폴더를 확인하세요.")
print("  ├ eda.png")
print("  ├ baseline_cm.png")
print("  ├ training_curve.png")
print("  ├ bert_cm.png")
print("  ├ model_comparison.csv")
print("  └ error_analysis.csv")
