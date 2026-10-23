# Bundle-RecSys

基于 RecBole 1.2.1 的推荐实验代码，当前提供 SASRec / SASRecCPR 在 Amazon Reviews 2023 四个领域上的训练、Top-K 推荐导出，以及目标用户评论和商品元数据提取流程。

仓库包含框架代码、配置、处理脚本和小型测试 fixtures。真实数据集、模型权重、日志与实际推荐列表保存在本地，不随仓库分发。当前流程是序列推荐；项目名称不代表已经实现独立的 bundle 推荐模型。

上游来源、基础 commit 与修改范围见 [UPSTREAM.md](UPSTREAM.md)。原 RecBole 文档保存在 [README_UPSTREAM.md](README_UPSTREAM.md)、[README_UPSTREAM_CN.md](README_UPSTREAM_CN.md) 和 `docs/`，使用本项目时以本文的路径和命令为准。

## 安装

使用 Python 3.11。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c constraints-tested.txt
python -m pip install --no-deps -e .
```

如只使用 CPU，可在安装其余依赖前安装 CPU 版 PyTorch：

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

运行依赖要求 NumPy < 2、Pandas < 3，以兼容上游配置和数据处理代码。Python 包和导入名仍为 `recbole`；其他上游模型需要的额外依赖按需安装。已验证的本地依赖版本和检查范围见 [docs/PROJECT_VALIDATION.md](docs/PROJECT_VALIDATION.md)。

## 数据准备

支持的领域如下。原始评论文件与 `meta_*.jsonl.gz` 需自行取得并放在本地。

| RecBole 数据集名 | 原始评论文件 | 商品元数据文件 |
| --- | --- | --- |
| `Amazon_All_Beauty` | `All_Beauty.jsonl.gz` | `meta_All_Beauty.jsonl.gz` |
| `Amazon_Appliances` | `Appliances.jsonl.gz` | `meta_Appliances.jsonl.gz` |
| `Amazon_Digital_Music` | `Digital_Music.jsonl.gz` | `meta_Digital_Music.jsonl.gz` |
| `Amazon_Health_and_Personal_Care` | `Health_and_Personal_Care.jsonl.gz` | `meta_Health_and_Personal_Care.jsonl.gz` |

推荐本地目录结构：

```text
dataset/                         # 被 Git 忽略
  All_Beauty.jsonl.gz
  meta_All_Beauty.jsonl.gz
  Amazon_All_Beauty/
    Amazon_All_Beauty.inter
saved/                           # 本地模型
outputs/recommendations/         # 本地推荐结果
```

将原始评论转换为 RecBole atomic 格式：

```bash
python prepare_amazon.py \
  --input dataset/All_Beauty.jsonl.gz \
  --dataset Amazon_All_Beauty \
  --dry-run

python prepare_amazon.py \
  --input dataset/All_Beauty.jsonl.gz \
  --dataset Amazon_All_Beauty
```

`--dry-run` 读取、校验和计数，不写入文件。转换脚本支持 JSONL 或 gzip JSONL；逐行使用 `user_id`、`parent_asin`、`rating`、`timestamp`，保留毫秒时间戳。每个“用户—商品”组合保留输入中最后一条记录，按保留记录的输入顺序输出；实际转换用临时 SQLite 文件完成去重，dry-run 使用内存数据库。评分和交互数量过滤由训练配置完成；已有输出不会被覆盖。输出表头为：

```text
user_id:token<TAB>item_id:token<TAB>rating:float<TAB>timestamp:float
```

脚本写入真实制表符，商品 ID 使用 `parent_asin`。其他领域替换文件名和数据集名即可。通过 `--output-root /path/to/datasets` 可指定外部数据目录，训练时相应设置 `--data_path=/path/to/datasets`。`data_path` 指数据集目录的父目录。

## 训练与评估

```bash
python run_recbole.py \
  --model=SASRec \
  --dataset=Amazon_All_Beauty \
  --config_files=configs/sasrec_amazon.yaml

python run_recbole.py \
  --model=SASRecCPR \
  --dataset=Amazon_All_Beauty \
  --config_files=configs/sasrec_amazon.yaml
```

两种模型共用配置，模型专属参数来自 `recbole/properties/model/`。配置采用 CE loss、关闭训练负采样、按时间排序和逐用户留一验证/测试，主验证指标为 NDCG@10。随机种子默认 2020，训练上限 500 epochs。可通过命令行覆盖，例如 `--epochs=1 --use_gpu=False` 做本地检查。

权重保存在 `saved/`，文件名包含模型、数据集和时间，如 `SASRec-Amazon-All-Beauty-<timestamp>.pth`。测试集汇总指标位于 `log/<model>/` 下日志的 `test result` 行；下面导出的推荐分数是逐用户结果，不是汇总指标。

## 导出 Top-K 推荐

以下命令中的 checkpoint 文件名需替换为本地实际文件名：

```bash
python predict_sasrec_topk.py \
  --checkpoint "saved/SASRec-Amazon-All-Beauty-<timestamp>.pth" \
  --top_users 20 \
  --topk 50 \
  --format markdown_compact \
  --output outputs/recommendations/sasrec_all_beauty_top20_top50.md
```

此处 `<timestamp>` 为占位符，使用时替换整个路径并用引号包裹。SASRecCPR 使用对应 checkpoint 和 `sasreccpr_` 输出前缀。也可使用 `--user USER_ID`、`--users_file users.txt`；支持 `markdown`、`markdown_compact` 和 `csv`。

脚本默认按测试样本的历史长度选取用户，调用模型 `full_sort_predict()`，屏蔽 padding 和输入序列中出现的商品。加载时需要 checkpoint 配置对应的原始交互数据，单独的权重文件不包含可移植数据集。仅加载自己生成或信任的 checkpoint。

## 提取用户评论和商品元数据

先为四个领域的两种模型各生成一个 `markdown_compact` 文件，文件名规则如下：

```text
{sasrec,sasreccpr}_{all_beauty,appliances,digital_music,health_and_personal_care}_top20_top50.md
```

以上为 8 个文件的简写，实际文件名不包含大括号。`top20_top50` 表示选取 20 位用户、每位用户保留 Top-50。生成到同一目录后执行：

```bash
python extract_amazon_target_user_content.py \
  --results-dir outputs/recommendations \
  --dataset-dir dataset \
  --output-dir dataset/amazon_target_user_content
```

脚本当前一次处理四个领域，每个领域合并两种模型结果中的目标用户。输出评论和元数据分别存储，保留原始 JSONL 记录。元数据覆盖“这些用户评论过的商品 ∪ 推荐列表中的商品”，并报告缺失的商品 ID 数量。为兼容旧命令，`--repo-dir` 仍是 `--results-dir` 的别名。

## 检查与测试

```bash
python -m pip install -r requirements-dev.txt
python scripts/verify_package.py
python -m pytest -q tests/project tests/metrics tests/data tests/config/test_config.py
```

检查脚本审核 Git 已跟踪和未忽略的文件，拒绝真实数据目录、权重、日志、推荐导出、大文件、机器绝对路径和可识别的凭据。真实运行前请先准备本地数据。`tests/` 下约 0.55 MiB 的小型测试 fixtures 用于离线回归；上游部分其他测试与示例需要另行准备数据或额外依赖。

## 许可证

遵循 [MIT License](LICENSE)，保留 RecBole 原始版权声明。使用 RecBole 及其模型开展研究时，请参阅上游文档中的论文引用。
