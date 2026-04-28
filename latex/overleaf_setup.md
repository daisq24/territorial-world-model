# Overleaf 上手指南 + NeurIPS Best Paper 学习清单

## A. 把这套 LaTeX 文件搬到 Overleaf 上（5 分钟）

### 路径 1：直接 Upload Project（最简单）

1. 在 Mac 上把 `latex/` 整个目录打成 zip：
   ```bash
   cd ~/Desktop/课程学习/25春\ 研二下课程/08\ 脑与机器智能/territorial_world_model
   zip -r tos_territorial_proposal.zip latex/ figures/
   ```
2. 登 [overleaf.com](https://www.overleaf.com) → New Project → Upload Project → 选 zip
3. Overleaf 会自动解压。打开 `latex/main.tex` 设为 main file，点 Recompile

### 路径 2：用 NeurIPS 官方模板（推荐，更专业）

1. 在 Overleaf 搜索框输入 `NeurIPS 2024` → 选官方模板 New Project
2. 删掉模板自带的 `main.tex` / `references.bib`，把我们的文件拖进去：
   - `latex/main.tex` → 项目根
   - `latex/sections/` 整个目录 → 项目根
   - `latex/refs.bib` → 项目根
   - `figures/` 整个目录 → 项目根
3. `main.tex` 顶部那段 `\IfFileExists{neurips_2025.sty}{...}` 会自动检测到模板里的 sty 并启用 NeurIPS 排版

### 路径 3：本地编译（备用）

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

需要的包：`amsmath`, `amssymb`, `amsthm`, `mathtools`, `bm`, `graphicx`,
`booktabs`, `caption`, `subcaption`, `algorithm`, `algpseudocode`,
`hyperref`, `cleveref`, `microtype`, `xspace`. 全是 TeX Live 标配。

---

## B. NeurIPS 2024 Best Papers（学习写作风格）

直接拿来读 introduction、relate work、experiments 怎么写——这几篇都是 5–10 页篇幅的范本，特别注意他们怎么用一句话把贡献说清楚。

| 奖项 | 论文 | arXiv | 学什么 |
|---|---|---|---|
| **Best Paper** | Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction (VAR) | [2404.02905](https://arxiv.org/abs/2404.02905) | 怎么把一个新表征讲清楚；图 1 用一张图说明白核心 idea |
| Runner-up | Stochastic Taylor Derivative Estimator (STDE) | [2412.00088](https://arxiv.org/abs/2412.00088) | **数学推导密集型论文的章节组织** —— 跟我们最像 |
| Runner-up | Not All Tokens Are What You Need for Pretraining (Rho-1) | [2404.07965](https://arxiv.org/abs/2404.07965) | 怎么用一个 selective 思路同时做实验+正当化 |
| Runner-up | Guiding a Diffusion Model with a Bad Version of Itself (Autoguidance) | [2406.02507](https://arxiv.org/abs/2406.02507) | **诊断型贡献怎么写** —— 跟我们的"territorial 是 ToS 的诊断扩展"非常对得上 |
| **D&B Best** | The PRISM Alignment Dataset | [2404.16019](https://arxiv.org/abs/2404.16019) | benchmark 类论文怎么motivate |

### 写作模仿要点

- **Abstract 第一句话**就把研究问题问出来（VAR 第一句："we present Visual AutoRegressive modeling..."；Autoguidance："we present a method..."）。我们的 abstract 已经按这个写了。
- **Introduction 用 3–4 段**：(1) 现状，(2) 关键 gap，(3) 我们的提议，(4) contributions 列表。看 Rho-1 的 intro。
- **Related Work 不要堆砌**：每段一个主题，每篇引一句话讲清楚。看 STDE 的相关工作。
- **Method 章节先 Definition 再 Algorithm**：跟我们的 §3 一致。看 STDE。
- **Limitations 要诚实写**：NeurIPS 现在强制要求。我们的 §6 / §8 已经写了。

---

## C. NeurIPS 2025 Best Papers（最前沿参考）

2025 共颁了 4 个 Best + 3 个 Runner-up，按 [NeurIPS Blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)：

| 奖项 | 论文 | 主题 |
|---|---|---|
| **Best Paper** | Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free (Alibaba Qwen team) | 注意力机制创新 |
| **Best Paper** | Artificial Hivemind / Infinity-Chat | 创造性多样性 benchmark |
| **Best Paper** | 1000 Layer Networks for Self-Supervised RL | 深度 RL |
| **Best Paper (D&B)** | (待 arXiv 公开；diffusion 训练动力学) | benchmark |
| Runner-ups (3) | online learning theory / LLM reasoning / scaling laws | — |

去 [neurips.cc/virtual/2025/awards_detail](https://neurips.cc/virtual/2025/awards_detail) 看 PDF。

> **特别注意 *Artificial Hivemind***：它跟我们这种"做诊断 + 提出 framework"的论文非常类似——介绍一个 benchmark + 揭示一个新现象，方法本身不复杂但思路清晰。是开题答辩可以模仿的"故事弧"。

---

## D. 我们这份 proposal 的 NeurIPS 风格 checklist

| 项 | 状态 |
|---|---|
| Abstract ≤ 250 词，第一句问出问题 | ✓ |
| Introduction 4 段（现状 / gap / 提议 / contributions） | ✓ |
| Related Work 4 个 paragraph，每段一个主题 | ✓ |
| Method 有 Definition + Algorithm + Proposition | ✓ |
| Experiments setup 章节单独写 | ✓ |
| Results 章节有 figure + table + 解读 | ✓（synthetic data，Day 2 替换为真数据） |
| Limitations / Risks 单独写 | ✓ |
| Future Work 2–3 条具体方向 | ✓ |
| Appendix 放完整证明 | ✓ |
| BibTeX 用 NeurIPS 风格 (`\bibliographystyle{plainnat}`) | ✓ |

这是个 **proposal-grade**（开题报告）骨架，不是 final paper。
真正投 NeurIPS 还要补：consistent results across seeds + 至少 2–3 个
foundation model 的 ablation + reproducibility checklist。

---

## E. 编译/上传时常见坑

1. **`neurips_2025.sty` not found** → 我的 main.tex 第 14–22 行做了 fallback，找不到就回到 `article + 1in margin`，编译不会失败。如果确实有 NeurIPS sty 但没启用，检查文件名拼写。
2. **`figures/fig01_metric_comparison.pdf` not found** → main.tex 用了 `\graphicspath{{figures/}{../figures/}}`，所以 zip 里 figures 跟 latex 同级或在 latex 上一级都能找到。Overleaf 上传后看 main.tex 在哪一层，调整 `\graphicspath`。
3. **bib 引用变成问号 `[?]`** → Recompile 一次后再点 `Logs and output files → bibtex` 跑 bibtex，再 Recompile 两次。Overleaf 现在一般会自动跑 bibtex，但偶尔要手动。
4. **中文字符（如 `脑与机器智能`）报错** → 我的 main.tex 没用中文。如果要在 acknowledgement 之类放中文，加 `\usepackage{ctex}` 并改用 XeLaTeX 编译。
