from argparse import Namespace
from datetime import datetime
from pathlib import Path

from help import get_default_model_path, list_models, set_default_model_path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "Models"

TRAIN_DEFAULTS = {
    "episodes": 30000,
    "batch_size": 128,
    "gamma": 0.99,
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "epsilon_start": 0.10,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.99995,
    "learning_starts": 1000,
    "update_target_every": 1000,
    "capacity_per_shape": 40000,
    "teacher_window": 150,
    "hidden_dim": 64,
    "res_blocks": 4,
    "log_every": 250,
    "save_every": 5000,
    "seed": 42,
    "cpu_threads": 4,
}

TRAIN_PROMPTS_NEW = [
    ("episodes", int, "训练局数"),
    ("batch_size", int, "batch_size"),
    ("gamma", float, "折扣因子 gamma"),
    ("lr", float, "学习率 lr"),
    ("weight_decay", float, "weight_decay"),
    ("epsilon_start", float, "初始探索率 epsilon_start"),
    ("epsilon_min", float, "最小探索率 epsilon_min"),
    ("epsilon_decay", float, "探索率衰减 epsilon_decay"),
    ("learning_starts", int, "开始学习前先积累多少步"),
    ("update_target_every", int, "Target 网络更新间隔"),
    ("capacity_per_shape", int, "每种 shape 的经验池容量"),
    ("teacher_window", int, "课程老师统计窗口"),
    ("hidden_dim", int, "隐藏通道数 hidden_dim"),
    ("res_blocks", int, "残差块数量 res_blocks"),
    ("log_every", int, "日志间隔(局)"),
    ("save_every", int, "保存间隔(局)"),
    ("seed", int, "随机种子"),
    ("cpu_threads", int, "CPU 线程数"),
]

TRAIN_PROMPTS_RESUME = [
    ("episodes", int, "继续训练多少局"),
    ("batch_size", int, "batch_size"),
    ("gamma", float, "折扣因子 gamma"),
    ("lr", float, "学习率 lr"),
    ("weight_decay", float, "weight_decay"),
    ("epsilon_min", float, "最小探索率 epsilon_min"),
    ("epsilon_decay", float, "探索率衰减 epsilon_decay"),
    ("learning_starts", int, "开始学习前先积累多少步"),
    ("update_target_every", int, "Target 网络更新间隔"),
    ("capacity_per_shape", int, "每种 shape 的经验池容量"),
    ("teacher_window", int, "课程老师统计窗口"),
    ("log_every", int, "日志间隔(局)"),
    ("save_every", int, "保存间隔(局)"),
    ("seed", int, "随机种子"),
    ("cpu_threads", int, "CPU 线程数"),
]

DIFFICULTIES = {
    "1": ("简单", {"width": 6, "height": 6, "mines": 6}),
    "2": ("普通", {"width": 8, "height": 8, "mines": 10}),
    "3": ("困难", {"width": 10, "height": 10, "mines": 18}),
}


def safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def prompt_value(label: str, default, cast_type):
    raw = safe_input(f"{label} [默认 {default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast_type(raw)
    except ValueError:
        print(f"输入无效，已使用默认值 {default}")
        return default


def print_models(models):
    current_default = get_default_model_path()
    current_default = str(Path(current_default).resolve()) if current_default else ""

    if not models:
        print("当前 Models 目录下没有找到模型文件。")
        return

    print("\n可用模型列表：")
    for idx, info in enumerate(models, start=1):
        compat_text = "兼容当前 DDQN" if info["compatible"] else f"旧格式/不兼容: {info['reason']}"
        cfg = info.get("model_config", {}) or {}
        cfg_text = ""
        if cfg:
            hidden_dim = cfg.get("hidden_dim")
            res_blocks = cfg.get("res_blocks")
            cfg_text = f" | hidden_dim={hidden_dim}, res_blocks={res_blocks}"
        ep_text = ""
        if info.get("episodes_finished"):
            ep_text = f" | 已训 {info['episodes_finished']} 局"
        default_tag = " <- 当前默认" if current_default and str(Path(info["path"]).resolve()) == current_default else ""
        print(f"  {idx}. {info['relative_path']} | {compat_text}{cfg_text}{ep_text}{default_tag}")


def choose_model_index(models, prompt_text="请选择模型编号（回车取消）: "):
    if not models:
        return None
    raw = safe_input(prompt_text).strip()
    if raw == "":
        return None
    if not raw.isdigit():
        print("请输入数字编号。")
        return None
    idx = int(raw) - 1
    if idx < 0 or idx >= len(models):
        print("编号超出范围。")
        return None
    return idx


def choose_model():
    models = list_models()
    print_models(models)
    idx = choose_model_index(models)
    if idx is None:
        return

    info = models[idx]
    if not info["compatible"]:
        print("这个模型与当前 Help / Practice / 续训流程不兼容，请选择兼容模型。")
        return

    saved_path = set_default_model_path(info["path"])
    print(f"已设置默认模型: {saved_path}")


def build_train_namespace(resume_from="", checkpoint_name="", metadata_name=""):
    values = dict(TRAIN_DEFAULTS)
    prompts = TRAIN_PROMPTS_RESUME if resume_from else TRAIN_PROMPTS_NEW

    print("\n直接回车即可使用默认参数。")
    for key, cast_type, label in prompts:
        values[key] = prompt_value(label, values[key], cast_type)

    values["checkpoint_name"] = checkpoint_name
    values["metadata_name"] = metadata_name
    values["resume_from"] = resume_from
    return Namespace(**values)


def train_model():
    models = list_models()
    print_models(models)
    mode = safe_input("\n1. 接着原来的某个模型继续训练\n2. 新建一个模型训练\n请选择 [默认 2]: ").strip() or "2"

    if mode == "1":
        idx = choose_model_index(models, "请选择要继续训练的模型编号（回车取消）: ")
        if idx is None:
            return

        info = models[idx]
        if not info["compatible"] or info["kind"] != "checkpoint":
            print("续训只支持当前兼容的 checkpoint 模型，请重新选择。")
            return

        checkpoint_name = Path(info["path"]).name
        metadata_name = Path(checkpoint_name).with_suffix(".json").name
        args = build_train_namespace(
            resume_from=info["path"],
            checkpoint_name=checkpoint_name,
            metadata_name=metadata_name,
        )
    else:
        default_name = f"minesweeper_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        filename = safe_input(f"请输入新模型文件名 [默认 {default_name}]: ").strip() or default_name
        if Path(filename).suffix == "":
            filename += ".pt"
        checkpoint_name = Path(filename).name
        metadata_name = Path(checkpoint_name).with_suffix(".json").name
        args = build_train_namespace(
            resume_from="",
            checkpoint_name=checkpoint_name,
            metadata_name=metadata_name,
        )

    print("\n即将使用如下训练参数：")
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")

    import train as train_module

    train_module.train(args)

    trained_path = MODELS_DIR / args.checkpoint_name
    if trained_path.exists():
        choice = safe_input("\n是否将这个模型设为默认模型？[Y/n]: ").strip().lower()
        if choice in ("", "y", "yes", "是"):
            set_default_model_path(str(trained_path))
            print(f"已将默认模型切换为: {trained_path}")


def choose_difficulty():
    print("\n请选择难度：")
    for key, (name, cfg) in DIFFICULTIES.items():
        print(f"  {key}. {name} ({cfg['width']}x{cfg['height']}, {cfg['mines']} 雷)")

    choice = safe_input("请选择难度编号 [默认 2]: ").strip() or "2"
    if choice not in DIFFICULTIES:
        print("输入无效，已自动选择普通难度。")
        choice = "2"
    return DIFFICULTIES[choice]


def play_game():
    _, cfg = choose_difficulty()
    model_path = get_default_model_path() or None

    try:
        import play as play_module

        play_module.main(
            width=cfg["width"],
            height=cfg["height"],
            no=cfg["mines"],
            block_size=40,
            model_path=model_path,
        )
    except Exception as exc:
        print(f"启动玩扫雷失败: {exc}")


def watch_ai():
    model_path = get_default_model_path()
    if not model_path:
        print("当前没有可用默认模型，请先选择模型。")
        return

    _, cfg = choose_difficulty()

    try:
        import practice as practice_module

        practice_module.enjoy(
            width=cfg["width"],
            height=cfg["height"],
            mines=cfg["mines"],
            model_path=model_path,
            step_delay=0.35,
        )
    except Exception as exc:
        print(f"启动 AI 扫雷演示失败: {exc}")


def print_menu():
    current_default = get_default_model_path()
    current_name = Path(current_default).name if current_default else "未设置"
    print("\n================ AI 扫雷主菜单 ================")
    print(f"当前默认模型: {current_name}")
    print("1. 选择模型")
    print("2. 训练模型")
    print("3. 玩扫雷")
    print("4. 看 AI 扫雷")
    print("0. 退出")


def main():
    while True:
        print_menu()
        choice = safe_input("请输入功能编号 [默认 0]: ").strip() or "0"

        try:
            if choice == "1":
                choose_model()
            elif choice == "2":
                train_model()
            elif choice == "3":
                play_game()
            elif choice == "4":
                watch_ai()
            elif choice == "0":
                print("已退出。")
                return
            else:
                print("无效选择，请重新输入。")
        except KeyboardInterrupt:
            print("\n已中断当前操作，返回主菜单。")


if __name__ == "__main__":
    main()
