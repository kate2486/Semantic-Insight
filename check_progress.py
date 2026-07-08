import sys, re, os, time, json
sys.stdout.reconfigure(encoding='utf-8')

output_path = r'C:\Users\ASUS\AppData\Local\Temp\claude\C--Users-ASUS-Desktop--------Semantic-Insight\e00b1b0b-7793-4b15-8c39-b7411b41de06\tasks\bdmiw60r4.output'
with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

clean = re.sub(r'\x1b\[[0-9;]*m', '', content)
clean = re.sub(r'\r', '\n', clean)

lines = [l.strip() for l in clean.split('\n') if l.strip()]

# 打印最后 80 行
print("=" * 60)
print("训练输出 (最后 80 行)")
print("=" * 60)
for line in lines[-80:]:
    print(line)

# 检查 checkpoint
print("\n" + "=" * 60)
print("Checkpoint 状态")
print("=" * 60)
ckpt = 'checkpoints/best_classification.pt'
if os.path.exists(ckpt):
    mtime = os.path.getmtime(ckpt)
    age = (time.time() - mtime) / 60
    print(f"保存时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
    print(f"距今: {age:.0f} 分钟前")
    print(f"文件大小: {os.path.getsize(ckpt)/1024/1024:.1f} MB")

# 检查训练历史
hist_path = 'checkpoints/training_history.json'
if os.path.exists(hist_path):
    print(f"\n训练历史文件: {hist_path}")
    with open(hist_path, 'r', encoding='utf-8') as f:
        hist = json.load(f)
    print(f"Train losses: {len(hist.get('train_loss', []))} epochs")
    print(f"Val losses: {len(hist.get('val_loss', []))} records")
    print(f"Val metrics: {[f'{m:.4f}' for m in hist.get('val_metric', [])]}")
    if 'test_accuracy' in hist:
        print(f"Test accuracy: {hist['test_accuracy']:.4f}")
