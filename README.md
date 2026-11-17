# Project 1：NumPy 复现 OpenCLIP ViT-B/32 图像编码器

郑贺元 2025681028 人工智能学院 人工智能专业 指导教师：樊超

## 项目内容

仅使用 NumPy 实现 OpenCLIP ViT-B/32 图像塔的完整前向计算：
输入预处理图像 `[B,3,224,224]`，输出 512 维归一化图像 token `[B,512]`，
与教师提供的独立 FP32 参考输出在容差内逐数值一致。

实现文件为 `image_encoder.py`（TODO-01 ~ TODO-06）：
patch 切分、多头注意力的头拆分与计算、Transformer 块（双残差 + MLP）、
token 嵌入、CLS 输出头。

## 运行环境

- Python 3.9+（本作业在 3.13 下验证通过）
- 仅需 numpy、pillow（运行课件示例需 imageio），无需 GPU / PyTorch / 网络

```bash
python -m pip install numpy pillow imageio
```

## 如何运行

```bash
# 分阶段数值校验（全部应显示 PASS）
python check.py --stage all

# 也可以逐阶段验证
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch

# 对附件自带的 10 张图像生成输出 token
python run.py --output output_tokens.npz

# 对任意自选图像生成输出 token
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

## 本次校验结果

全部 8 个阶段 PASS。输出阶段（10 张图，(10,512)）max_abs = 5.4e-07，
远小于容差 2e-05；batch 检验（乱序 + 重复图像）通过。

## 文件说明

- `image_encoder.py` —— 本人完成的实现（本次提交的核心文件）
- `报告.docx` —— 实验报告（基于老师下发的《可操作的实验报告模版》填写，共 4 页）
- 附件原有的 `helpers.py`、`check.py`、`run.py`、`weights/`、`assets/` 等未做任何修改
