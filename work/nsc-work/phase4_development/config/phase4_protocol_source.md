# CS5491 MB-NSC Project
# Phase 4: Development Experiment Protocol

**项目：** Module-Balanced Neural Spectral Capacity for Transformer Architecture Ranking

**当前阶段：** Phase 4 — Development Evaluation

**前置阶段：**
- Phase 2：数学规范完成。
- Phase 3：Python 实现完成，70 项测试通过。
- Phase 3.1：补充审计完成，90 项测试通过。

**本协议目的：** 在运行 Development Evaluation 前，固定评价指标、候选选择规则、实验顺序、数据隔离和结果记录格式。

---

## 1. 实验目标

本阶段研究：

Attention–FFN 容量均衡机制是否能够在 Original NSC 的基础上，为 Transformer 架构性能排序提供额外预测信息。

需要回答三个问题：

1. Layer-wise 与 Global 哪种聚合方式在 Development 上表现更好？
2. 两种模型分别采用哪个 lambda 时表现最好？
3. MB-NSC 是否能够改善排序相关性，以及固定参数预算下的架构选择效果？

本阶段不预先假定 MB-NSC 一定优于 Original NSC。

如果两个变体都没有改善基准结果，也必须如实报告。

## 2. 数据集与数据隔离

### 2.1 数据划分

使用课程提供的 200 个归档 Decoder-only Transformer 架构。

| 数据集 | 架构数量 | 标签 | 用途 |
|---|---:|---|---|
| Development | 100 | Validation PPL | 参数搜索、模型选择 |
| Final | 100 | Test PPL | 最终泛化验证 |

本阶段只使用 Development。

### 2.2 严格隔离规则

允许读取：
- `architectures.json`
- Development 对应的架构配置。
- `development_labels.json`
- Development 所需的资源信息。
- 官方评估器。

禁止读取：
- `final_labels.json`
- Final Test PPL。
- Final 预测结果。
- 任何 Final 排名或性能统计。

禁止：
- 根据 Final 结果选择模型。
- 根据 Final 结果修改 lambda。
- 根据 Development 排名反向修改数学公式。
- 使用架构 ID 或配置指纹查表。
- 在评分函数中读取 PPL 标签。

如果发现 Development 标签缺失、重复或存在非法值，必须先报告，不得静默删除样本。

预期 Development 样本数必须为 100。

## 3. 评价指标

### 3.1 主要指标：Spearman's Rank Correlation

**<span>Spearman's ρ（斯皮尔曼等级相关系数）</span>**

将 MB-NSC 的架构评分与负 Validation PPL 进行比较。

\[
\rho=\operatorname{Spearman}(S,-PPL)
\]

其中：
- \(S\)：模型计算的架构评分。
- \(PPL\)：Development Validation Perplexity。
- \(-PPL\)：负困惑度，数值越大表示性能越好。

选择 Spearman 作为主要指标的原因：

1. 本项目核心目标是架构排序。
2. Spearman 衡量预测排名与实际性能排名之间的一致程度。
3. 能直接衡量模型是否提供额外的排序信息。
4. 与 Proposal 的研究目标一致。

**选模原则：Spearman ρ 越高越好。**

### 3.2 次要指标：Kendall's Tau-b

\[
\tau_b=\operatorname{Kendall}(S,-PPL)
\]

Kendall τ_b 用于衡量两个排序之间的一致性，并考虑并列排名。

主要用途：

- 检查 Spearman 的结论是否得到另一种排序指标支持。
- 分析模型评分出现并列时的影响。
- 作为主要指标并列时的第二级比较依据。

Spearman 是主要指标，Kendall τ_b 是次要指标。

不得在实验完成后，根据哪个指标更有利而交换二者的优先级。

### 3.3 评价指标汇总

| 指标 | 作用 | 优先级 |
|---|---|---|
| Spearman ρ | 整体架构排序相关性 | 第一 |
| Kendall τ_b | 排序一致性及并列处理 | 第二 |
| Size-controlled pairwise analysis | 控制架构规模差异 | 补充 |
| Budgeted selection / Regret | 实际资源预算下的选择效果 | 补充 |

### 3.4 排名并列处理

严格沿用官方 `evaluate.py` 的架构评分比较规则：

- 使用 12 位有效数字进行评分比较。
- Spearman 使用平均秩处理并列。
- Kendall 使用 τ_b 的并列修正。
- 不在 `student_score.py` 内提前舍入分数。

注意：

官方评分的 12 位有效数字规则，不等于模型选择时的指标并列阈值。

两者必须分开处理。

对于模型评价指标之间的并列，本协议另行规定：

\[
|\rho_i-\rho_j|\leq10^{-12}
\]

视为 Spearman 并列。

对于 Kendall：

\[
|\tau_{b,i}-\tau_{b,j}|\leq10^{-12}
\]

视为 Kendall 并列。

该容差只用于候选模型的选模比较，不改变架构评分或正式数学公式。

常量评分向量导致相关系数为 null 时：

- 记录为 undefined/null。
- 不得擅自改为 0。
- 不得使用 Kendall 替代无定义的 Spearman。
- 不得将 null 候选作为有效的最佳模型。
- 如果所有候选的主要指标均为 null，停止选模并报告问题。

---

## 4. 超参数搜索方案

### 4.1 固定搜索网格

严格使用 Phase 2 已确定的 lambda 网格：

\[
\Lambda=
\{0,0.1,0.2,0.3,0.4,0.5,
0.6,0.7,0.8,0.9,1.0\}
\]

每个模型 11 个候选值。

| 模型 | lambda 候选数量 |
|---|---:|
| Layer-wise Quadratic MB-NSC | 11 |
| Global Quadratic MB-NSC | 11 |
| 合计 | 22 |

不得：
- 改变步长。
- 增加新的 lambda 候选。
- 删除表现不好的候选。
- 根据中间结果重新设计搜索网格。
- 为其中一个模型增加调参机会。

lambda=0 是 Original NSC Recovery Control，必须保留在两个模型的候选记录中。

### 4.2 公平比较原则

两个模型必须满足：

1. 使用完全相同的 Development 架构。
2. 使用相同的 Development 标签。
3. 使用相同的评价指标。
4. 使用相同的资源预算。
5. 使用相同的搜索网格。
6. 使用相同的官方评估器。
7. 使用相同的并列处理方式。

每个模型只能使用自己的 11 个候选值进行搜索。

不能根据某个模型的结果，为另一个模型增加额外候选。

---

## 5. 最终模型选择规则

这是本协议新增的核心决策。

采用两阶段、分层选择。

### 5.1 第一阶段：分别选择每个模型的最佳 lambda

Layer-wise：

\[
\lambda_L^*=
\arg\max_{\lambda\in\Lambda}
\rho_L(\lambda)
\]

Global：

\[
\lambda_G^*=
\arg\max_{\lambda\in\Lambda}
\rho_G(\lambda)
\]

其中：
- \(\rho_L\)：Layer-wise 在 Development 上的 Spearman。
- \(\rho_G\)：Global 在 Development 上的 Spearman。

每个模型独立从自己的 11 个候选中选择。

### 5.2 第二阶段：比较两个模型的最佳候选

比较：

- Layer-wise + 最佳 lambda。
- Global + 最佳 lambda。

采用以下固定优先级：

**第一优先级：Spearman ρ 最大。**

**第二优先级：如果 Spearman 并列，选择 Kendall τ_b 更高的候选。**

**第三优先级：如果两个指标均并列，选择 lambda 较小的候选。**

较小 lambda 意味着对 Original NSC 的修改程度较弱，可作为保守的预先规定的平分规则。

**第四优先级：如果两个模型仍然完全并列，选择 Global。**

原因：Global 的聚合方式更简单，不需要保留逐层惩罚结果。

必须注明：这只是完全并列情况下的操作性选择，不代表 Global 在预测性能上优于 Layer-wise。

### 5.3 lambda=0 的特殊解释

如果最终选出的候选为：

\[
\lambda=0
\]

则：

\[
S_L=S_G=S_0
\]

意味着关闭均衡机制的原始 NSC 已经达到该选模规则下的最佳表现。

此时不得宣称 MB-NSC 的均衡机制改善了预测效果。

应如实得出：

**在当前 Development 存档中，没有足够证据表明引入正强度的 Attention–FFN 均衡惩罚能够改善排序表现。**

不得强制选择正数 lambda。

### 5.4 补充指标不参与主要选模

以下结果必须报告，但不参与主要模型选择规则：

- Size-controlled pairwise results。
- Top-1 / Top-3 selection。
- Regret。
- Random baseline。
- 资源基线比较。

这样可以避免在多个指标中事后挑选对某个模型最有利的结果。

---

## 6. Size-controlled Pairwise Analysis

### 6.1 比较条件

沿用 Proposal 中的限制：

1. 两个架构必须具有相同 hidden width。
2. 参数量满足：

\[
\frac{\min(P_i,P_j)}
{\max(P_i,P_j)}
\geq0.90
\]

其中 \(P_i,P_j\) 为架构总参数量。

Development 预期比较数量：

\[
N_{\mathrm{pairs}}=180
\]

必须核对实际数量是否为 180。

如果数量不一致，先检查官方 pair 构造方式，不得擅自改变条件以凑足 180。

### 6.2 结果记录

对这组固定架构对，记录：

- 符合实际 PPL 顺序的预测对数。
- 预测顺序相反的对数。
- 预测评分并列的对数。
- 实际 PPL 并列的对数。
- 实际有效比较对数。
- Pairwise concordance accuracy。

若官方评估器已经提供相应统计量，优先保留其原始定义和输出，不得改变课程评估器的口径。

如果需要补充独立的 pairwise accuracy，则定义：

\[
Acc_{\mathrm{pair}}=
\frac{N_{\mathrm{correct}}+
0.5N_{\mathrm{predicted\ ties}}}
{N_{\mathrm{valid\ pairs}}}
\]

其中：
- 正确预测顺序记 1。
- 预测并列记 0.5。
- 预测错误记 0。
- 实际 PPL 并列的对不计入有效比较分母，单独报告。

注意：180 个架构对存在重叠，不是 180 个独立统计样本。

Size-controlled 结果用于检查控制规模差异后的排序表现，不能直接宣称已经识别出容量均衡机制的因果效果。

---

## 7. Budgeted Architecture Selection

### 7.1 参数预算

严格使用 Proposal 规定的四个参数上限：

| Parameter Cap | k |
|---|---|
| 10M | 1、3 |
| 20M | 1、3 |
| 30M | 1、3 |
| 50M | 1、3 |

总共八种预算选择场景。

可行架构必须满足：

\[
P_{\mathrm{total}}\leq B
\]

其中 B 为当前参数预算。

使用 `resources.json` 中的 `total` 字段。

不能将可能重叠的参数组件字段全部相加代替官方总参数。

### 7.2 Top-1 / Top-3 选择

对每种模型和基线：

1. 筛选符合参数预算的架构。
2. 按模型评分从高到低排序。
3. 选择排名前 k 的架构。
4. 读取对应 Development PPL。
5. 记录所选架构及其实际 PPL。

当 k=3 时，记录三个架构的 PPL，并计算其中最低的 PPL。

如果可行架构数量不足 k：
- 如实记录实际可选数量。
- 不得补造架构。
- 不得将不存在的候选视为有效选择。

如果没有可行架构，则该场景标记为 N/A。

### 7.3 Regret

定义：

\[
Regret=
PPL_{\mathrm{selected\ best}}
-
PPL_{\mathrm{feasible\ best}}
\]

其中：

- \(PPL_{\mathrm{selected\ best}}\)：模型选出的 Top-k 架构中最低的 PPL。
- \(PPL_{\mathrm{feasible\ best}}\)：当前参数预算下所有可行架构的最低 PPL。

Regret 越低越好。

理想情况：

\[
Regret=0
\]

表示选出的架构中包含该预算下实际表现最好的架构。

### 7.4 随机基线

使用 200 次固定随机种子试验。

每次试验必须：
- 使用相同的可行架构池。
- 使用相同参数预算。
- 使用相同 k。
- 执行均匀随机选择。

记录：

- 平均 Best PPL。
- 中位数 Best PPL。
- 标准差。
- 平均 Regret。
- 中位数 Regret。

随机基线的标准差代表随机选择的不确定性，不是模型训练种子的不确定性。

架构评分并列导致 Top-k 边界不确定时，也必须沿用官方评估器的固定种子随机平分处理规则，并记录相应结果。

不得使用架构 ID 作为隐藏的平分优先级。

---

## 8. 必须比较的基线

所有方法必须使用相同的 Development 数据与预算条件。

| 方法 | 用途 |
|---|---|
| Original NSC | 原始基准 |
| Positional Example | Starter 提供的位置示例，作为 sanity check |
| Layer-wise MB-NSC | 11 个 lambda 候选 |
| Global MB-NSC | 11 个 lambda 候选 |
| MB-NSC Ablation | 关闭均衡机制的控制 |
| Parameter Count | 参数量资源基线 |
| Reference FLOPs | 参考计算量基线 |
| Uniform Random | 随机选择基线 |

### 8.1 Reference FLOPs 口径

投影参数量：

\[
P_{\mathrm{proj}}
=
\sum_l(4d^2+2df_l)
\]

参考 MACs：

\[
MACs_{\mathrm{token}}
=
P_{\mathrm{proj}}+2LKd
\]

其中：

\[
K=384
\]

参考 FLOPs：

\[
FLOPs_{\mathrm{reference}}
=2MACs_{\mathrm{token}}
\]

该指标是课程规定的参考计算量估计，不等同于真实推理延迟，也不代表完整训练 FLOPs。

按官方 Starter 实际实现的资源基线方向和计算方式运行，不得擅自更改资源定义。

---

## 9. 实验执行顺序

严格按照以下顺序进行。

### Step 0：实验前检查

确认：

- Phase 3 和 Phase 3.1 全部测试通过。
- 两个模型均可通过官方接口运行。
- lambda 网格包含 11 个值。
- 两个模型均使用同一网格。
- Development 和 Final 数据划分正确。
- 代码中没有读取标签的逻辑。
- 实验输出目录不会覆盖现有文件。

记录：
- Python 版本。
- 操作系统。
- CPU 型号。
- 运行环境。
- 当前代码文件 SHA-256。
- 当前 Git commit（如果已有）。
- 实际运行命令。

### Step 1：Development Baseline

先运行并记录：

- Original NSC。
- Positional Example。
- MB-NSC Ablation。
- Parameter Count。
- Reference FLOPs。

核对：

\[
S_{\mathrm{Ablation}}=S_{\mathrm{Original}}
\]

如果不一致，立即停止后续实验并检查代码。

### Step 2：Layer-wise 搜索

依次运行：

\[
\lambda=0.0,0.1,\ldots,1.0
\]

共 11 次候选评估。

每个候选必须使用固定的进程级模型配置，不能在同一进程中导入后再擅自切换模型参数。

### Step 3：Global 搜索

使用完全相同的网格：

\[
\lambda=0.0,0.1,\ldots,1.0
\]

共 11 次候选评估。

### Step 4：汇总 22 个候选

生成：

- Spearman ρ。
- Kendall τ_b。
- Size-controlled pairwise results。
- 八种参数预算下的 Top-1 / Top-3。
- Regret。

所有 22 个候选必须保留完整记录。

### Step 5：随机基线与资源基线

完成 200 次固定种子随机选择。

同时整理 Parameter Count、Reference FLOPs 等基线。

### Step 6：Development 模型选择

按照第 5 节规定的优先级：

1. Spearman。
2. Kendall τ_b。
3. 更小 lambda。
4. 完全并列时选择 Global。

分别确定两个模型的最佳 lambda，再比较两个模型。

不得改变选择规则。

### Step 7：生成 Phase 4 报告

完成全部表格、图表、运行日志、结论和限制说明。

**本阶段结束后暂停，不得自动运行 Final Evaluation。**

---

## 10. 实验结果记录格式

必须保存原始结果，不得只保留最佳候选。

### 10.1 22 个候选的排名结果表

文件：

`candidate_ranking_results.csv`

| Model | Lambda | Spearman ρ | Kendall τ_b | Size-controlled result | Status |
|---|---:|---:|---:|---:|---|
| Layer-wise | 0.0 | | | | |
| Layer-wise | 0.1 | | | | |
| Layer-wise | ... | | | | |
| Layer-wise | 1.0 | | | | |
| Global | 0.0 | | | | |
| Global | 0.1 | | | | |
| Global | ... | | | | |
| Global | 1.0 | | | | |

必须保留全部 22 行。

### 10.2 模型选择结果表

文件：

`model_selection_results.csv`

| Model | Best Lambda | Spearman ρ | Kendall τ_b | Selection Rank |
|---|---:|---:|---:|---:|
| Layer-wise | | | | |
| Global | | | | |

同时记录：
- 第一阶段选模过程。
- 是否发生指标并列。
- 是否使用次级规则。
- 最终选定的模型。
- 最终选定的 lambda。
- 选择原因。

### 10.3 Budgeted Selection 结果表

文件：

`budgeted_selection_results.csv`

| Method | Budget | k | Selected Architectures | Selected Best PPL | Feasible Best PPL | Regret |
|---|---:|---:|---|---:|---:|---:|
| Original NSC | 10M | 1 | | | | |
| Original NSC | 10M | 3 | | | | |
| Layer-wise | 10M | 1 | | | | |
| Global | 10M | 1 | | | | |
| ... | ... | ... | | | | |

必须覆盖四种预算、两种 k 和所有预定方法。

### 10.4 Random Baseline 结果表

文件：

`random_baseline_results.csv`

| Budget | k | Trials | Mean Best PPL | Median Best PPL | Std | Mean Regret |
|---|---:|---:|---:|---:|---:|---:|
| 10M | 1 | 200 | | | | |
| 10M | 3 | 200 | | | | |
| 20M | 1 | 200 | | | | |
| ... | ... | ... | | | | |

### 10.5 Size-controlled 结果表

文件：

`size_controlled_results.csv`

| Method | Valid Pairs | Correct | Incorrect | Predicted Ties | Actual Ties | Pairwise Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Original NSC | | | | | | |
| Layer-wise | | | | | | |
| Global | | | | | | |

必须保留实际 pair 数量及官方定义。

### 10.6 运行记录

文件：

`run_manifest.csv`

至少记录：

- Model。
- Lambda。
- Python 版本。
- 运行命令。
- 开始时间。
- 结束时间。
- 运行耗时。
- 输出文件。
- 代码版本或哈希。
- 成功或失败状态。
- 错误信息。

MP 缓存可能影响运行计时。不得将共享缓存条件下的运行时间直接解释为模型冷启动速度优势。

---

## 11. 结果可视化

至少生成以下图表：

### Figure 1：Spearman ρ 与 lambda 的关系

- 横轴：lambda。
- 纵轴：Development Spearman ρ。
- 两条曲线：Layer-wise、Global。
- 标注各自最佳 lambda。

### Figure 2：Kendall τ_b 与 lambda 的关系

- 横轴：lambda。
- 纵轴：Development Kendall τ_b。
- 两条曲线：Layer-wise、Global。

### Figure 3：与 Original NSC 的排序性能比较

展示：
- Original NSC。
- Layer-wise 最佳候选。
- Global 最佳候选。
- Parameter Count。
- Reference FLOPs。
- Positional Example。

分别展示 Spearman 和 Kendall。

### Figure 4：Budgeted Selection Regret

- 横轴：参数预算。
- 纵轴：Regret。
- 区分 k=1 和 k=3。
- 比较各方法及随机基线。

### Figure 5：Size-controlled Pairwise Comparison

比较 Original NSC、Layer-wise、Global 在固定 180 对中的表现。

不得只选择有利图表而省略其他预定结果。

---

## 12. Phase 4 文件输出目录

建议在工作副本中建立独立结果目录：

`work/nsc-work/phase4_development/`

建议结构：

```text
phase4_development/
    config/
        protocol_snapshot.json

    raw_results/
        candidate_ranking_results.csv
        model_selection_results.csv
        budgeted_selection_results.csv
        random_baseline_results.csv
        size_controlled_results.csv

    logs/
        run_manifest.csv
        layerwise/
        global/

    figures/
        figure1_spearman.png
        figure2_kendall.png
        figure3_ranking_comparison.png
        figure4_regret.png
        figure5_size_controlled.png

    summary/
        development_summary.json
```

在 analysis 中生成：

`MB_NSC_Phase4_Development_Report.md`

不要覆盖：
- 原始 Starter。
- Phase 2 数学规范。
- Phase 3 实施报告。
- Phase 3.1 审计报告。
- 现有 JSON 数据及历史实验结果。

---

## 13. Development 结论撰写要求

报告必须回答：

1. Layer-wise 的最佳 lambda 是多少？
2. Global 的最佳 lambda 是多少？
3. 哪个模型取得更高的 Spearman？
4. Kendall 是否支持相同结论？
5. 与 Original NSC 相比，是否有排序提升？
6. Size-controlled 结果是否支持整体结论？
7. 固定预算下，MB-NSC 是否降低 Regret？
8. 两种模型在哪些参数预算下表现较好？
9. 是否存在某些 lambda 使评分退化？
10. 均衡机制是否提供了超出总容量的信息？

必须区分：

- 机制敏感度证据。
- Development 排序相关性证据。
- Budgeted Selection 实用性证据。

不能把其中一种证据代替另外两种。

### 结果解释限制

Development 包含 100 个架构，而 22 个候选均在该数据集上进行选择。

因此，最佳 Development 结果是经过调参后的选择结果，不是独立泛化性能的无偏估计。

180 个规模控制配对存在重叠，不得当作独立样本。

不得根据单一归档数据集声称模型已经具有跨数据集泛化能力。

如果 MB-NSC 未超过 Original NSC，应如实报告负面结果。

---

## 14. Final 阶段隔离

Phase 4 完成后：

1. 确认最终模型。
2. 确认最终 lambda。
3. 冻结数学规范。
4. 冻结 Python 实现。
5. 冻结评估协议。
6. 记录代码版本、Git commit 或 tag。
7. 保存完整 Development 实验结果。

之后才允许进入 Final。

Final 只运行：

- Development 已选定的模型及 lambda。
- Original NSC。
- Proposal 规定的固定基线。

Final 不再运行 22 个候选进行选模。

Final 结果不得反过来修改模型或 lambda。

---

## 15. Phase 4 完成标准

必须同时满足：

- 22 个候选全部完成评估。
- 两个模型均有完整 11 点结果。
- Original NSC 基线已完成。
- Ablation 与 Original NSC 一致。
- Spearman 和 Kendall 均有记录。
- Size-controlled 分析完成。
- 四种预算、两种 k 的选择结果完成。
- 200 次随机基线完成。
- Regret 计算完成。
- 所有原始结果和运行日志保存。
- 最终模型及 lambda 按预注册规则确定。
- Development 报告生成。
- Final 标签未被读取。

如果任何候选运行失败，不得静默跳过。

如果发生代码错误：
1. 停止当前批次。
2. 记录错误和已有运行结果。
3. 修复代码。
4. 重新进行 Phase 3/3.1 必要的回归测试。
5. 以同一协议重新运行全部 22 个候选。
6. 不得混合不同代码版本的结果进行选模。

**Phase 4 完成后，必须向项目负责人提交完整报告并等待审核，不得自动进入 Final。**

