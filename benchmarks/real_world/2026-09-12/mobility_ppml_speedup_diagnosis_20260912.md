# PPML-HDFE 加速效果诊断

> **公开版说明**：本文件为公开分发做了脱敏，原先指向本机绝对安装路径的链接已改写为包内相对模块路径；引用行号、技术结论与全部数值均未改动。

## 结论

当前六年 PPML benchmark 中，econhdfe `optimized` 的整体加速比约 3.74x，`replica` 约 2.24x。主要原因不是 PPML IRLS 收敛慢，而是这份数据的 separation 结构非常重，同时 econhdfe 的 direct PPML API 没有真正传递 `ExecutionConfig` 的线程和内存参数。

## 1. 2014 年隔离测试

2014 年每年输入 1,166,400 行，最终保留 499,833 行，666,567 行被分离删除，占输入样本 57.1%。完整规格与只启用 FE separation 的结果如下：

| 配置 | 传统 `ppmlhdfe`（秒） | econhdfe optimized（秒） | 加速比 |
|---|---:|---:|---:|
| 默认：`fe + simplex + relu` | 153.945 | 37.781 | 4.07x |
| 只启用：`fe` | 72.552 | 9.670 | 7.50x |
| 额外 separation 检查成本 | 81.393 | 28.111 | — |

econhdfe optimized 的阶段计时：

| 阶段 | 秒 | 占完整耗时 |
|---|---:|---:|
| FE separation | 0.013 | 0.03% |
| simplex separation | 7.337 | 19.42% |
| ReLU separation | 20.617 | 54.57% |
| separation 合计 | 28.061 | 74.27% |
| IRLS | 6.660 | 17.63% |
| 列筛选 + VCE 等 | 约 2.05 | 约 5.4% |

`simplex` 和 `relu` 在该年都返回 0 个新增分离样本；也就是说，它们是为一般情形保留的正确性检查，但在该数据上主要体现为额外计算成本。

## 2. 与 Stata 的比较

本机安装的 `ppmlhdfe` 默认也使用 `fe simplex relu` 三类 separation 方法。因此默认配置的比较是同一估计目标和同一类稳健性检查，并非 econhdfe 单方面多做了检查。

但这也解释了为什么总体加速比没有达到复杂三维 FE benchmark 的 10 倍以上：两边都要在每年 116.64 万行上处理大量零流量和分离样本。各年份分离比例为 57.1%–79.0%，因此相当一部分时间花在“确定哪些观测不能进入 PPML 估计”，而不是花在最终保留样本上的 IRLS。

## 3. 代码层面的性能问题

### 3.1 PPML direct API 没有传递 `ExecutionConfig`

`econhdfe/models/ppml/api.py` 的 `ppmlhdfe()` 只校验 `execution_config`，随后直接调用 `fit_arrays()`，没有把 `threads`、`memory_budget_mb`、`cache` 传进去。对应的 `econhdfe/models/ppml/api.py:53` 和 `econhdfe/hdfe/weighted_projection.py:22` 中，projector 实际使用的是 `absorb_threads="auto"`。

因此原 benchmark 文件名中的“4 threads”并不等于 PPML 内部实际使用 4 个 Numba 线程；在未设置环境变量时，本机实际为 8 个线程。受内存带宽限制，2014 年诊断中 4 线程 optimized 为 37.78 秒，8 线程为 38.49 秒；4 线程 replica 为 61.03 秒，8 线程为 61.80 秒。这个问题会影响严格的 apples-to-apples 比较，但不是 3.74x 加速不高的主因。

同一原因也意味着 benchmark 中传入的 2048 MB 内存预算没有直接控制 PPML direct API 的内部 projector。诊断运行的进程峰值 RSS 约 4.2 GB，需在修复参数传递后重新记录。

### 3.2 optimized 引擎没有覆盖 separation 阶段

`mixed_simplex_separation()` 内部把正交化调用硬编码为 `engine="replica"`，见 `econhdfe/models/ppml/separation_simplex.py:136`。所以即便主估计选择 optimized，simplex 仍使用 replica 路径。

ReLU 路径则固定创建 `method="lsmr"` 的 HDFE absorber，见 `econhdfe/models/ppml/separation_relu.py:33`，没有使用 optimized 主估计中的两路 Schur/PCG projector。它还会在 ReLU 迭代中反复 residualize 边界向量，因此在本数据上产生约 20.6 秒成本。

## 4. 不是主要问题的部分

- 外层 IRLS 迭代数与 Stata 完全一致：各年为 11–13 次。
- replica 的 HDFE 子迭代数与 Stata 基本一致，例如 2014 年均为 189 次；这说明不是 econhdfe 因收敛标准过严导致外层重复迭代。
- VCE 成本很小，2014 年约 0.16 秒。
- 2014 年 FE-only 后样本、迭代数和系数仍与完整配置一致；因此当前隔离测试没有改变估计结果。

## 5. 修复优先级

1. 先让 PPML direct API 把 `ExecutionConfig.threads` 和内存策略传递到 projector；随后重新跑严格 4-thread benchmark。
2. 将 separation 各方法的执行计划和引擎参数统一传入，至少消除 simplex 的硬编码 replica 路径。
3. 对 ReLU / simplex 增加“结果为空但成本很高”的阶段记录，并评估基于 FE separation 结果的安全短路或复用 residualization 结构。
4. 修复后分别报告“完整默认 separation”与“只启用 FE separation”的性能；后者只能作为敏感性/上界，不应替代默认 PPML 的正式结果。

本诊断使用的脚本和原始诊断 CSV 均保存在本 benchmark 输出目录下，未修改源数据、源复现脚本或已安装包文件。
