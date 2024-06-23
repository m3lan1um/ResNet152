import json
import matplotlib.pyplot as plt

# JSON 파일 로드
with open('results/training_results.json', 'r') as f:
    results = json.load(f)

train_losses = results['train_losses']
train_accuracies = results['train_accuracies']
val_losses = results['val_losses']
val_accuracies = results['val_accuracies']

epochs = range(1, len(train_losses) + 1)

# Validation accuracy가 가장 높은 에폭 찾기
best_epoch = val_accuracies.index(max(val_accuracies)) + 1
print(f'Best epoch: {best_epoch}')
print(f'Training Loss: {train_losses[best_epoch - 1]}')
print(f'Training Accuracy: {train_accuracies[best_epoch - 1]}')
print(f'Validation Loss: {val_losses[best_epoch - 1]}')
print(f'Validation Accuracy: {val_accuracies[best_epoch - 1]}')

# 플롯 생성
plt.figure(figsize=(12, 5))

# 손실 플롯
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.axvline(best_epoch, color='red', linestyle='--', label='Best Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

# 정확도 플롯
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.axvline(best_epoch, color='red', linestyle='--', label='Best Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()

# 플롯을 파일로 저장
plt.tight_layout()
plt.savefig('results/training_results_plot.png')
# plt.show()
