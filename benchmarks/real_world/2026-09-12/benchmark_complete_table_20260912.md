# econhdfe 完整 benchmark 表格

> 生成时间：2026-09-12。所有脚本、日志和结果均位于 benchmark 输出目录；原始数据和原始复现脚本只读使用，未被覆盖。
>
> **公开版说明**：本文件为公开分发做了脱敏，原数据集的本地项目名称已泛化为 `Project-A` / `Project-B`；未发表项目的真实名称不在公开版本中。全部数值结果、样本量、口径说明均未改动。

## 一、总表

| 数据集 | benchmark | 规格 | 计时口径 | 回归/年份数 | 传统引擎 | 传统耗时（秒） | econhdfe 耗时（秒） | 加速比 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| Project-A | primary | OLS，共同可比单元 | 12 个共同可比回归合计 | 12 | reghdfe | 75.250 | 9.659 | 7.79x | pass |
| Project-A | primary | IV，共同可比单元 | 6 个共同可比回归合计 | 6 | ivreghdfe | 162.107 | 29.481 | 5.50x | pass |
| Project-A | primary | OLS + IV，共同可比单元 | 18 个共同可比回归合计 | 18 | reghdfe + ivreghdfe | 237.357 | 39.140 | 6.06x | pass |
| Project-B | PPML-HDFE | 六年 PPML，econhdfe optimized 引擎 | 6 个年份回归 fit timer 合计 | 6 | ppmlhdfe | 800.036 | 213.681 | 3.74x | pass |
| Project-B | PPML-HDFE | 六年 PPML，econhdfe replica 引擎 | 6 个年份回归 fit timer 合计 | 6 | ppmlhdfe | 800.036 | 357.788 | 2.24x | pass |
| Project-A | complex 3-way FE | ACF：firm + year + city×year | 单个全样本回归 fit timer | 1 | reghdfe | 686.525 | 58.708 | 11.69x | pass |
| Project-A | complex 3-way FE | ACF：firm + year + city×year | 单个全样本回归 fit timer | 1 | reghdfejl（独立进程） | 326.491 | 58.708 | 5.56x | pass |
| Project-A | complex 3-way FE | ACF：firm + year + city×year | 单个全样本回归 fit timer | 1 | reghdfejl（同进程 warm） | 346.448 | 58.708 | 5.90x | pass |
| Project-A | complex 3-way FE | OP：firm + year + city×year | 单个全样本回归 fit timer | 1 | reghdfe | 701.413 | 58.130 | 12.07x | pass |
| Project-A | complex 3-way FE | OP：firm + year + city×year | 单个全样本回归 fit timer | 1 | reghdfejl（独立进程） | 364.936 | 58.130 | 6.28x | pass |
| Project-A | complex 3-way FE | OP：firm + year + city×year | 单个全样本回归 fit timer | 1 | reghdfejl（同进程 warm） | 377.412 | 58.130 | 6.49x | pass |

## 二、Project-A：primary OLS / IV 明细

| 模型 | 类型 | 固定效应 | 共同回归数 | 样本 N | 传统合计（秒） | econhdfe 合计（秒） | 加速比 | N 匹配 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| OLS export ratio (ACF) | OLS | firm_id + year | 3 | 1352317 | 25.375 | 3.338 | 7.60x | 是 | pass |
| OLS export ratio (OP) | OLS | firm_id + year | 3 | 1352317 | 25.660 | 3.124 | 8.21x | 是 | pass |
| OLS export value (ACF) | OLS | firm_id + year | 1 | 455078 | 2.775 | 0.354 | 7.83x | 是 | pass |
| OLS export value (OP) | OLS | firm_id + year | 1 | 455078 | 2.771 | 0.331 | 8.36x | 是 | pass |
| OLS exporter status (ACF) | OLS | firm_id + year | 1 | 1352317 | 7.542 | 1.050 | 7.19x | 是 | pass |
| OLS exporter status (OP) | OLS | firm_id + year | 1 | 1352317 | 7.634 | 1.034 | 7.38x | 是 | pass |
| OLS forward continuation (ACF) | OLS | firm_id + year | 1 | 261639 | 1.747 | 0.208 | 8.40x | 是 | pass |
| OLS forward continuation (OP) | OLS | firm_id + year | 1 | 261639 | 1.746 | 0.219 | 7.97x | 是 | pass |
| IV preferred (ACF) | IV | firm_id + year | 3 | 1352317 | 80.807 | 12.613 | 6.41x | 是 | pass |
| IV preferred (OP) | IV | firm_id + year | 3 | 1352317 | 81.300 | 16.868 | 4.82x | 是 | pass |

## 三、Project-B PPML 六年明细

| 年份 | econhdfe 引擎 | Stata N | econhdfe N | ppmlhdfe（秒） | econhdfe（秒） | 加速比 | 分离样本数 | 迭代数 | N 匹配 | 分离数匹配 | |loglike 差| | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2010 | optimized | 245116 | 245116 | 136.400 | 37.131 | 3.67x | 921284 | 13 | 是 | 是 | 5.02e-08 | pass |
| 2012 | optimized | 317704 | 317704 | 127.645 | 35.955 | 3.55x | 848696 | 12 | 是 | 是 | 2.54e-07 | pass |
| 2014 | optimized | 499833 | 499833 | 153.945 | 41.573 | 3.70x | 666567 | 12 | 是 | 是 | 5.23e-07 | pass |
| 2016 | optimized | 487203 | 487203 | 132.522 | 29.523 | 4.49x | 679197 | 11 | 是 | 是 | 4e-07 | pass |
| 2018 | optimized | 443547 | 443547 | 120.327 | 32.968 | 3.65x | 722853 | 12 | 是 | 是 | 6.25e-08 | pass |
| 2020 | optimized | 436857 | 436857 | 129.197 | 36.530 | 3.54x | 729543 | 12 | 是 | 是 | 1.93e-07 | pass |
| 2010 | replica | 245116 | 245116 | 136.400 | 60.101 | 2.27x | 921284 | 13 | 是 | 是 | 8.72e-08 | pass |
| 2012 | replica | 317704 | 317704 | 127.645 | 59.173 | 2.16x | 848696 | 12 | 是 | 是 | 2.62e-07 | pass |
| 2014 | replica | 499833 | 499833 | 153.945 | 70.018 | 2.20x | 666567 | 12 | 是 | 是 | 5.18e-07 | pass |
| 2016 | replica | 487203 | 487203 | 132.522 | 57.111 | 2.32x | 679197 | 11 | 是 | 是 | 3.93e-07 | pass |
| 2018 | replica | 443547 | 443547 | 120.327 | 52.177 | 2.31x | 722853 | 12 | 是 | 是 | 5.63e-08 | pass |
| 2020 | replica | 436857 | 436857 | 129.197 | 59.208 | 2.18x | 729543 | 12 | 是 | 是 | 1.91e-07 | pass |

## 四、复杂三维固定效应明细

| 指标 | 引擎 | 样本 N | 迭代数 | 耗时（秒） | 相对 econhdfe 加速比 | 交互项系数 | 标准误 | 状态 |
|---|---|---|---|---|---|---|---|---|
| ACF | econhdfe | 1352122 | 2633 | 58.708 | 1.00x | -0.00835681 | 0.00262290 | pass |
| OP | econhdfe | 1352122 | 2633 | 58.130 | 1.00x | -0.01297762 | 0.00291824 | pass |
| ACF | reghdfejl（独立进程） | 1352122 | 2051 | 326.491 | 5.56x | -0.00835678 | 0.00262309 | pass |
| OP | reghdfejl（独立进程） | 1352122 | 2051 | 364.936 | 6.28x | -0.01297766 | 0.00291846 | pass |
| ACF | reghdfe | 1352122 | 1148 | 686.525 | 11.69x | -0.00835681 | 0.00262290 | pass |
| OP | reghdfe | 1352122 | 1148 | 701.413 | 12.07x | -0.01297763 | 0.00291824 | pass |
| ACF | reghdfejl（同进程 warm） | 1352122 | 2051 | 346.448 | 5.90x | -0.00835678 | 0.00262309 | pass |
| OP | reghdfejl（同进程 warm） | 1352122 | 2051 | 377.412 | 6.49x | -0.01297766 | 0.00291846 | pass |

## 五、口径与解释

1. primary trade benchmark 的共同可比部分是 18 个回归：12 个 OLS + 6 个 IV。完整执行中 Stata 侧实际跑了 30 个 fit，econhdfe 侧实际跑了 18 个 fit；总表的 primary 行只比较共同可比的 18 个。
2. PPML 的 800.036 秒是六个 `ppmlhdfe` 命令的 fit timer 合计，不是整个 do-file 的墙钟时间；Stata 包装脚本墙钟约 13 分 37 秒。项目原始正式日志约 22 分 14 秒还包含数据构造和结果写盘，因此不与 fit timer 直接混比。
3. PPML 六年中 econhdfe replica 引擎合计 357.788 秒、整体加速约 2.24x；optimized 引擎合计 213.681 秒、整体加速约 3.74x。
4. 复杂三维规格使用数据中实际存在的 `firm_id + year + cityyear_id`（city×year）。该面板没有 `industry_id` 或 `industry×year` 变量，因此没有擅自把估计对象改成 city×year×industry×year。
5. 三维规格中，`reghdfejl（独立进程）` 是修复后的独立进程结果；`reghdfejl（同进程 warm）` 作为启动后同进程敏感性结果单列，不与独立进程基准混为一谈。所有三维全样本结果均返回 rc=0。

## 六、配套 CSV

- 总表：`benchmark_complete_summary_20260912.csv`
- Trade primary 明细：`benchmark_complete_trade_primary_detail_20260912.csv`
- PPML 六年明细：`benchmark_complete_mobility_ppml_detail_20260912.csv`
- 复杂三维 FE 明细：`benchmark_complete_complex_3way_detail_20260912.csv`
