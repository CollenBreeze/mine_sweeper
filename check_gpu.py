import torch
import sys
import os


def diagnostic_report():
    print("=" * 30)
    print("🚀 扫雷 AI - GPU 深度诊断报告")
    print("=" * 30)

    # 1. 软件环境检查
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否编译进 PyTorch: {torch.cuda.is_available()}")

    # 2. 驱动与硬件识别
    if torch.cuda.is_available():
        print(f"✅ GPU 识别成功!")
        print(f"当前 GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"当前 CUDA 版本: {torch.version.cuda}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 2:.2f} MB")

        # 3. 实时运算测试（核心步骤）
        print("\n--- 正在进行算力握手测试 ---")
        try:
            # 创建两个大矩阵并在 GPU 上做乘法
            a = torch.randn(2000, 2000).to("cuda")
            b = torch.randn(2000, 2000).to("cuda")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            c = torch.matmul(a, b)
            end.record()

            torch.cuda.synchronize()
            print(f"✨ 矩阵乘法测试完成，耗时: {start.elapsed_time(end):.2f} ms")
            print("结论: 你的 GPU 正在全速工作！")
        except Exception as e:
            print(f"❌ 运算测试失败: {e}")

    else:
        # 4. 失败原因溯源
        print("❌ GPU 不可用，排查建议：")

        # 检查是否有 NVIDIA 显卡
        if os.system("nvidia-smi") != 0:
            print("- 错误: 系统未发现 NVIDIA 驱动。请安装或更新显卡驱动。")
        else:
            print("- 驱动已安装，但 PyTorch 无法调用。")
            print("- 可能原因: 你安装的是 CPU 版的 PyTorch，或者 CUDA 版本与驱动不匹配。")
            print("- 建议尝试重新执行: pip install torch --index-url https://download.pytorch.org/whl/cu121")

    print("=" * 30)


if __name__ == "__main__":
    diagnostic_report()