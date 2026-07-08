"""
训练进度监控脚本
定期检查 checkpoint 更新时间并在训练完成后运行测试集评估
"""
import os
import sys
import time
import json
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

CHECKPOINT = "checkpoints/best_classification.pt"
REPORT_PATH = r"C:\Users\ASUS\Desktop\神经网络实训\训练验证指标报告.md"


def get_checkpoint_info():
    """获取 checkpoint 信息"""
    if not os.path.exists(CHECKPOINT):
        return None
    return {
        "mtime": os.path.getmtime(CHECKPOINT),
        "size_mb": os.path.getsize(CHECKPOINT) / 1024 / 1024,
    }


def print_status():
    """打印当前训练状态"""
    info = get_checkpoint_info()
    if info:
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["mtime"]))
        age_min = (time.time() - info["mtime"]) / 60
        print(f"[{time.strftime('%H:%M:%S')}] 最佳模型: {mtime_str} ({age_min:.0f} 分钟前) | 大小: {info['size_mb']:.1f} MB")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] 尚未保存 checkpoint")


def main():
    print("=" * 60)
    print("训练监控 — 每 5 分钟检查一次")
    print("按 Ctrl+C 退出")
    print("=" * 60)

    prev_mtime = None
    while True:
        info = get_checkpoint_info()
        if info and info["mtime"] != prev_mtime:
            print(f"\n🆕 新 checkpoint! ({time.strftime('%H:%M:%S', time.localtime(info['mtime']))})")
            prev_mtime = info["mtime"]

        print_status()

        # 检查训练进程是否还在运行
        import subprocess
        result = subprocess.run(
            'tasklist /FI "IMAGENAME eq python.exe" /FO CSV 2>NUL | find /c "python.exe"',
            shell=True, capture_output=True, text=True
        )
        python_count = int(result.stdout.strip() or 0)

        if python_count < 2:
            print("\n⚠ 训练进程可能已结束！")
            break

        time.sleep(300)  # 5 分钟


if __name__ == "__main__":
    main()
