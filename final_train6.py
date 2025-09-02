import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler

from preprocess import get_data_loaders

# AMP import (PyTorch 2.x 권장 → 구버전 폴백)
try:
    from torch.amp import autocast, GradScaler   # PyTorch 2.x
    AMP_NS = "torch.amp"
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    AMP_NS = "torch.cuda.amp"

# 재현성 고정
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True
cudnn.benchmark = False  # 고정 크기라도 deterministic 우선

# 전역 최적화 플래그
try:
    torch.set_float32_matmul_precision("high")   # Ampere/ADA 최적화
except Exception:
    pass

# 경로
base_dir = "/workspace/datasets"
train_dir = os.path.join(base_dir, "train")
test_dir  = os.path.join(base_dir, "test")

# 하이퍼파라미터
num_classes   = 2
num_epochs    = 100
batch_size    = 16
learning_rate = 1e-5
weight_decay  = 1e-4
label_smooth  = 0.03           # 0.05 -> 0.03 (과스무딩 완화)

# 평가/ES 옵션 (NO-TTA만 사용)
NEG_CLASS_INDEX  = 0
THR_GRID         = np.linspace(0.30, 0.70, 9)
EVAL_EVERY_EPOCH = 1
WARMUP_ES        = 5           # ES는 5에폭 이후부터
PATIENCE_ES      = 12          # 8 -> 12 (개선 여지 확보)
MIN_DELTA        = 1e-4        # 개선 최소 폭
RESUME_FROM_BEST = True        # acc 최고 체크포인트 이어달리기 허용
RESUME_LR        = 3e-6        # 이어달리기 시 LR

# 디바이스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_enabled = (device.type == "cuda")
print(f"현재 디바이스: {device} | AMP: {AMP_NS} (enabled={amp_enabled})")

# 데이터 (preprocess.py에서 320x320 + 얼굴정렬/보수적 증강)
train_loader, test_loader = get_data_loaders(
    train_dir, test_dir, image_size=(320, 320), batch_size=batch_size
)

# 클래스 불균형: WeightedRandomSampler 적용
def _get_targets_from_imagefolder(dataset):
    if hasattr(dataset, "targets") and len(dataset.targets):
        return np.array(dataset.targets)
    return np.array([s[1] for s in dataset.samples])

train_dataset = train_loader.dataset
labels_np = _get_targets_from_imagefolder(train_dataset)
class_counts = np.bincount(labels_np, minlength=2)

weights_per_class = 1.0 / np.maximum(class_counts, 1)
sample_weights = weights_per_class[labels_np]
sampler = WeightedRandomSampler(
    torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True
)

# DataLoader 재구성(셔플 대신 sampler)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=train_loader.num_workers,
    pin_memory=True,
    persistent_workers=(train_loader.num_workers > 0),
    collate_fn=train_loader.collate_fn,
)

# 모델
weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
# Dropout 강화 (기본 0.2/0.3 → 0.5)
if isinstance(model.classifier, nn.Sequential) and isinstance(model.classifier[0], nn.Dropout):
    model.classifier[0] = nn.Dropout(p=0.5)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)
model.to(memory_format=torch.channels_last)

# 손실/옵티마이저/스케줄러
# CE + class weight(역빈도) + label smoothing
w0 = 1.0 / max(int(class_counts[0]), 1)  # alert(neg index=0)
w1 = 1.0 / max(int(class_counts[1]), 1)  # non-alert
# alert 쪽 살짝 상향(×1.1) -> F1(neg) 급락 방지
w0 = float(w0) * 1.1
class_weight = torch.tensor([w0, w1], dtype=torch.float32, device=device)
class_weight = class_weight / class_weight.mean()  # 평균 1로 정규화
print(f"class_counts={class_counts.tolist()}, class_weight={class_weight.tolist()}")

criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smooth)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# CosineAnnealingWarmRestarts: 후반 미세개선 허용
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=learning_rate * 0.1
)

scaler = GradScaler(enabled=amp_enabled)

def sweep_thresholds(y_true, p_neg, thresholds):
    """
    y_true: [N] (0=alert, 1=non-alert)  ※ 여기서 'neg'는 alert 클래스(인덱스 0)
    p_neg : [N] (alert 확률; alert을 '양성'으로 간주)
    """
    y_true = np.asarray(y_true)
    p_neg  = np.asarray(p_neg)

    rows = []
    best_by_acc   = (None, -1.0)
    best_by_f1neg = (None, -1.0)

    auc_neg = None
    try:
        auc_neg = roc_auc_score(y_true == 0, p_neg)
    except Exception:
        pass

    for thr in thresholds:
        y_pred = (p_neg >= thr).astype(np.int64)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], average=None, zero_division=0
        )
        prec_neg, rec_neg, f1_neg = prec[0], rec[0], f1[0]
        rows.append((thr, acc, prec_neg, rec_neg, f1_neg))
        if acc > best_by_acc[1]:
            best_by_acc = (thr, acc)
        if f1_neg > best_by_f1neg[1]:
            best_by_f1neg = (thr, f1_neg)

    print("\n[Threshold sweep on alert probability]")
    if auc_neg is not None:
        print(f"AUC (alert as positive): {auc_neg:.4f}")
    print(" thr   |  acc    |  alert-prec  alert-rec   alert-f1")
    print("-----------------------------------------------------")
    for thr, acc, p0, r0, f10 in rows:
        print(f" {thr:0.2f} | {acc:0.4f} |  {p0:0.4f}      {r0:0.4f}     {f10:0.4f}")
    print("\nBest by ACC     : thr={:.2f}, acc={:.4f}".format(*best_by_acc))
    print("Best by F1(alert): thr={:.2f}, f1_alert={:.4f}".format(*best_by_f1neg))

    return {
        "table": rows,
        "best_by_acc": best_by_acc,
        "best_by_f1neg": best_by_f1neg,
        "auc_neg": auc_neg,
    }

# 체크포인트에서 이어달리기, 선택사항임
if RESUME_FROM_BEST and os.path.exists("sun_train5.pth"):
    try:
        state = torch.load("sun_train5.pth", map_location=device)
        model.load_state_dict(state)
        for g in optimizer.param_groups:
            g['lr'] = RESUME_LR
        print(f"[Resume] Loaded sun_train5.pth and set LR={RESUME_LR:g}")
    except Exception as e:
        print("[Resume] 실패:", e)

# 학습 루프
best_acc = 0.0
best_f1_neg = -1.0
best_auc = -1.0
stale = 0

all_preds = []
all_labels = []

# (AUC/ROC 계산용) 마지막 에폭의 확률 저장
probs_neg_no_last = []
labels_no_last = []

for epoch in range(num_epochs):
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")

    # Train
    model.train()
    total_train_loss = 0.0
    correct, total = 0, 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        if images.numel() == 0:  # robust_collate 방어
            continue
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = (correct / total) if total > 0 else 0.0
    print(f"Train Loss: {total_train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Eval (NO-TTA)
    model.eval()
    total_test_loss = 0.0
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    probs_neg_no  = []
    labels_no     = []

    with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            if images.numel() == 0:
                continue
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            _, predicted_argmax = torch.max(outputs, dim=1)
            correct += (predicted_argmax == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted_argmax.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            # NO-TTA 확률 (alert=NEG_CLASS_INDEX)
            prob_no = F.softmax(outputs, dim=1)[:, NEG_CLASS_INDEX]
            probs_neg_no.extend(prob_no.detach().cpu().numpy())
            labels_no.extend(labels.detach().cpu().numpy())

    test_acc = (correct / total) if total > 0 else 0.0
    print(f"Test Loss: {total_test_loss:.4f}, Test Acc (argmax, no TTA): {test_acc:.4f}")

    # 임계값 스윕 & 조기종료 판단
    if (epoch + 1) % EVAL_EVERY_EPOCH == 0:
        print("\n[NO-TTA threshold sweep]")
        res_no = sweep_thresholds(labels_no, probs_neg_no, THR_GRID)
        # 모니터링 지표
        curr_f1neg = res_no["best_by_f1neg"][1] if res_no["best_by_f1neg"][0] is not None else -1.0
        curr_auc = res_no["auc_neg"] if res_no["auc_neg"] is not None else -1.0

        print(f"\n[Suggest] NO-TTA thr (F1 alert): {res_no['best_by_f1neg'][0]:.2f}")

        # 체크포인트 저장: F1(alert) 최고
        if curr_f1neg > best_f1_neg + MIN_DELTA:
            best_f1_neg = curr_f1neg
            torch.save(model.state_dict(), "sun_train6_f1.pth")
            print("Best (F1 alert) model saved as sun_train6_f1.pth")

        # 정확도 최고 시점 저장 + ES patience 리셋
        if test_acc > best_acc + MIN_DELTA:
            best_acc = test_acc
            torch.save(model.state_dict(), "sun_train6.pth")
            print("Best model saved (acc).")
            stale = 0  # 정확도 개선이면 patience 리셋

        # ES 판단 (웜업 이후, 지표 이중화: F1 또는 AUC가 개선되면 리셋)
        if (epoch + 1) >= WARMUP_ES:
            improved = False
            if curr_f1neg > best_f1_neg + MIN_DELTA:
                improved = True
            if curr_auc > best_auc + MIN_DELTA:
                improved = True

            if improved:
                best_f1_neg = max(best_f1_neg, curr_f1neg)
                best_auc = max(best_auc, curr_auc)
                stale = 0
            else:
                stale += 1
                print(f"[EarlyStopping] stale={stale}/{PATIENCE_ES} "
                      f"(best F1_alert={best_f1_neg:.4f}, best AUC={best_auc:.4f})")
                if stale >= PATIENCE_ES:
                    print(">> Early stopping triggered.")
                    # 스케줄러는 루프 외부에서도 안전하지만 여기서 한 번 더
                    scheduler.step(epoch + 1)
                    # AUC/ROC 출력을 위해 마지막 값 보존
                    probs_neg_no_last = probs_neg_no
                    labels_no_last = labels_no
                    break

    # 스케줄러 스텝 (WarmRestarts는 epoch로 호출)
    scheduler.step(epoch + 1)

    # AUC/ROC 출력을 위해 마지막 루프 값 갱신
    probs_neg_no_last = probs_neg_no
    labels_no_last = labels_no
else:
    # 정상 종료(for-else): break 없이 끝난 케이스
    pass

# 최종 리포트
print("\nTraining complete.")
print(f"Best Test Accuracy (argmax, no TTA): {best_acc:.4f}")
print("\n[Confusion Matrix]")
if len(all_labels) and len(all_preds):
    print(confusion_matrix(all_labels, all_preds))
    print("\n[Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=['alert', 'non-alert']))
else:
    print("N/A (no predictions collected)")

# AUC / ROC 출력
try:
    y_true = np.array(labels_no_last)
    y_score = np.array(probs_neg_no_last)  # 마지막 에폭 no-TTA alert 확률
    if len(y_true) and len(y_score):
        auc = roc_auc_score(y_true == 0, y_score)
        fpr, tpr, thresholds = roc_curve(y_true == 0, y_score)
        print(f"\n[ROC AUC] (alert as positive): {auc:.4f}")
        print("FPR:", fpr)
        print("TPR:", tpr)
        print("Thresholds:", thresholds)
    else:
        print("\n[ROC AUC] 계산 불가: 저장된 확률/라벨이 비어 있음")
except Exception as e:
    print("\nROC AUC 계산 중 오류:", e)
