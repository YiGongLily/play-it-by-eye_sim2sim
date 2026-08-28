# X2 Depth CNN Sim2Sim (MuJoCo + viser)

[English](README.md) | [中文](README_zh.md)

[**Play It By Eye**](https://github.com/YiGongLily/play-it-by-eye) 的配套 MuJoCo 闭环仓库。

## 概述

在 MuJoCo + viser 中闭环运行 **LeggedLab Depth Rough-CNN** X2 策略的 sim2sim。

- **不修改** [Play It By Eye](https://github.com/YiGongLily/play-it-by-eye) 训练代码；使用普通 Python venv（无需 Isaac Lab conda 环境）。
- 策略链路：**Play It By Eye 导出 ONNX → 旁路 `deploy_real` → 本仓库**。

## 前置

请将依赖仓库与本仓库放在同一父目录下：

| 仓库 | 作用 |
|------|------|
| [play-it-by-eye](https://github.com/YiGongLily/play-it-by-eye) | 训练 / 导出 ONNX |
| `deploy_real` | ONNX 包、`deployment_alignment.yaml`、部署配置 |
| `x2_rl_deploy` | MuJoCo 机器人 mesh（见 `assets/x2_with_camera.xml` 的 `meshdir`） |

默认布局：

```text
workspace/
├── play-it-by-eye/          # 本地也可能叫 leggedlab
├── deploy_real/
├── x2_rl_deploy/
└── play-it-by-eye_sim2sim/  # 本仓库（本地也可能叫 x2_depth_sim2sim）
```

若 mesh 路径不同，请按相对该 XML 的路径修改 `assets/x2_with_camera.xml` 中的 `meshdir`。

完整 sim2sim 流程见 [Play It By Eye](https://github.com/YiGongLily/play-it-by-eye) 文档。

## 安装

```bash
cd /path/to/play-it-by-eye_sim2sim   # 或你的本地目录名
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 验收

```bash
.venv/bin/python scripts/smoke_ort.py    # pass=True
.venv/bin/python scripts/smoke_loop.py   # 无界面烟雾
```

## 运行（viser）

```bash
# 平地
.venv/bin/python run_sim2sim.py --viser-port 8080

# 楼梯 + 斜坡（viser 右侧 Terrain Difficulty 切换 Easy / Medium / Hard）
.venv/bin/python run_sim2sim.py --config configs/sim2sim_stairs.yaml --viser-port 8080
```

浏览器打开终端打印的 URL；右侧面板调节速度与地形难度。

## 楼梯难度

| 难度 | 楼梯 | 坡度 | 出生 y |
|------|------|------|--------|
| Easy | 4×8 cm | 0.20 | 0 |
| Medium | 5×12 cm | 0.25 | 5 |
| Hard | 5×20 cm | 0.38 | 10 |
