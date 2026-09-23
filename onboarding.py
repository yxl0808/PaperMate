"""First-run onboarding flow."""
import os
from pathlib import Path
from config import DEEPSEEK_API_KEY


def run_onboarding() -> bool:
    """Run onboarding if first use. Returns True if ready to proceed."""
    data_dir = Path(__file__).parent / "data"
    if not os.path.exists(data_dir / "learner.db"):
        print("\n" + "=" * 50)
        print("  欢迎使用 PaperMate！")
        print("=" * 50)
        print("\n看起来你还没有导入任何论文。\n")

        if not DEEPSEEK_API_KEY:
            print("⚠ 未检测到 DEEPSEEK_API_KEY。")
            print("  请在 .env 文件中配置:\n")
            print("  DEEPSEEK_API_KEY=sk-your-key-here")
            print("\n  配置完成后重新启动。")
            return False

        print("初始化步骤:")
        print("  1. ✅ DEEPSEEK_API_KEY 已配置")
        print("  2. 📁 将论文 PDF 放入 data/papers/ 目录")
        print("  3. 📥 选择菜单 [1] 导入新论文\n")
        print("直接输入论文路径即可导入:\n")

    return True
