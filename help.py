import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from Models.ddqn import DDQN


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "Models"
APP_CONFIG_PATH = PROJECT_ROOT / "app_config.json"
SUPPORTED_MODEL_SUFFIXES = {".pt", ".pth"}
PREFERRED_DEFAULT_MODEL = MODELS_DIR / "minesweeper_anyshape_ddqn.pt"


def _normalize_model_path(model_path: Optional[str]) -> Path:
    if not model_path:
        return Path("")
    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _to_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _read_app_config() -> Dict:
    if not APP_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_app_config(config: Dict):
    APP_CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def inspect_model_file(model_path: str) -> Dict:
    path = _normalize_model_path(model_path)
    info = {
        "name": path.name if path.name else str(model_path),
        "path": str(path),
        "relative_path": _to_relative_path(path) if path.name else str(model_path),
        "compatible": False,
        "kind": "unknown",
        "model_config": {},
        "episodes_finished": None,
        "reason": "",
    }

    if not path.exists():
        info["reason"] = "文件不存在"
        return info

    try:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "model_state_dict" in payload:
            model_cfg = payload.get("model_config", {})
            model = DDQN(**model_cfg)
            model.load_state_dict(payload["model_state_dict"])
            info.update(
                {
                    "compatible": True,
                    "kind": "checkpoint",
                    "model_config": model_cfg,
                    "episodes_finished": payload.get("episodes_finished"),
                }
            )
            return info

        if isinstance(payload, dict):
            model = DDQN()
            model.load_state_dict(payload)
            info.update(
                {
                    "compatible": True,
                    "kind": "state_dict",
                    "model_config": {"hidden_dim": model.hidden_dim, "res_blocks": model.res_blocks},
                }
            )
            return info

        info["reason"] = "不是可识别的 PyTorch 模型文件"
        return info
    except Exception as exc:
        text = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
        info["reason"] = text[:160]
        return info


def list_models() -> List[Dict]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_infos: List[Dict] = []
    for path in sorted(MODELS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES:
            model_infos.append(inspect_model_file(str(path)))
    return model_infos


def set_default_model_path(model_path: str) -> str:
    path = _normalize_model_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")

    config = _read_app_config()
    config["default_model_path"] = _to_relative_path(path)
    _write_app_config(config)
    return str(path)


def get_default_model_path() -> str:
    config = _read_app_config()
    configured_path = config.get("default_model_path", "")
    if configured_path:
        path = _normalize_model_path(configured_path)
        if path.exists():
            return str(path)

    if PREFERRED_DEFAULT_MODEL.exists():
        info = inspect_model_file(str(PREFERRED_DEFAULT_MODEL))
        if info["compatible"]:
            set_default_model_path(str(PREFERRED_DEFAULT_MODEL))
            return str(PREFERRED_DEFAULT_MODEL.resolve())

    for info in list_models():
        if info["compatible"]:
            set_default_model_path(info["path"])
            return str(Path(info["path"]).resolve())

    return ""


def load_model(model_path: Optional[str] = None, device: Optional[torch.device] = None):
    resolved_path = _normalize_model_path(model_path) if model_path else _normalize_model_path(get_default_model_path())
    if not resolved_path or not resolved_path.exists():
        raise FileNotFoundError("未找到可用的默认模型，请先在 main.py 中选择模型。")

    model_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = DDQN.from_checkpoint(str(resolved_path), map_location=model_device)
    model = model.to(model_device)
    model.eval()
    return model, payload, str(resolved_path)


class Help:
    def __init__(self, model_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self._resolve_model_path(model_path)
        self.model = None
        self.payload = None

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        resolved = _normalize_model_path(model_path) if model_path else _normalize_model_path(get_default_model_path())
        if not resolved or not resolved.exists():
            raise FileNotFoundError("未找到可用模型，请先在 main.py 中设置默认模型。")
        return str(resolved)

    def ensure_model(self):
        if self.model is None:
            self.model, self.payload, self.model_path = load_model(self.model_path, self.device)
        return self.model

    def best_action(self, state, mask, board_shape=None):
        model = self.ensure_model()

        state_arr = np.asarray(state, dtype=np.float32)
        mask_arr = np.asarray(mask, dtype=np.float32)
        if state_arr.ndim != 2 or mask_arr.ndim != 2:
            raise ValueError("Help.best_action 需要二维盘面 state 和 mask")

        if state_arr.shape != mask_arr.shape:
            raise ValueError(f"state.shape={state_arr.shape} 与 mask.shape={mask_arr.shape} 不匹配")

        board_shape = tuple(board_shape) if board_shape is not None else tuple(state_arr.shape)
        valid_actions = np.flatnonzero(mask_arr.reshape(-1) > 0.5)
        if len(valid_actions) == 0:
            return None

        state_t = torch.as_tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_t, mask_t, board_shape=board_shape).squeeze(0).detach().cpu().numpy()

        action_idx = int(np.argmax(q_values))
        row = action_idx // board_shape[1]
        col = action_idx % board_shape[1]
        return {
            "row": int(row),
            "col": int(col),
            "action_idx": action_idx,
            "score": float(q_values[action_idx]),
            "model_path": self.model_path,
        }

    def best_action_from_env(self, env):
        state = env.state.astype(np.float32).copy()
        mask = env.get_mask(flatten=False).astype(np.float32)
        return self.best_action(state, mask, board_shape=(env.grid_height, env.grid_width))
