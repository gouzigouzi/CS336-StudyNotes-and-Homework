# CS336 Assignment 4 (data): Filtering Language Modeling Data

## 4 排行榜：为语言建模筛选数据
现已实现多种网络爬取数据筛选基元（primitives），接下来将其应用于生成语言建模训练数据。

本部分任务目标：筛选 Common Crawl（CC）WET 文件集合，生成语言建模训练数据。我们已在 Together 集群的 /data/CC/ 目录下提供 5000 个 WET 文件（文件格式为 CC*.warc.wet.gz）作为数据来源。

具体目标：筛选 CC 数据 dump，生成语言建模数据，使得基于该数据训练的 Transformer 语言模型在 Paloma 基准测试的 C4 100 domains 子集上的验证困惑度（perplexity）最小化。不得修改模型架构或训练流程，核心目标是构建最优数据。该验证集包含 C4 语言建模数据集中 100 个最常见域名的样本（参考 Raffel 等人 2020 年的研究），我们已在 Together 集群的 `/data/paloma/` 目录下提供该数据的分词版本（使用 GPT-2 分词器处理），文件路径为：`/data/paloma/tokenized_paloma_c4_100_domains_validation.bin`。可通过以下代码加载并查看数据：
```python
import numpy as np
data = np.fromfile(
    "/data/paloma/tokenized_paloma_c4_100_domains_validation.bin",
    dtype=np.uint16
)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.decode(data[0:2000]))  # 解码前2000个token查看数据
```
基于筛选后的数据集，需在该数据上训练一个 GPT-2 small 规格的模型（200K迭代次数），并在 C4 100 验证集上评估其困惑度。

注意事项：允许利用 Paloma 验证集构建筛选器或分类器以处理 CC WET 文件，但严禁将验证集数据直接复制到训练数据中，语言模型不得接触任何验证集数据。

5000 个 WET 文件的数据量相当可观，压缩后约 375GB。为高效处理数据，建议尽可能使用多进程并行计算。Python 的`concurrent.futures`或`multiprocessing` API可能会有所帮助。以下是使用 `concurrent.futures` 实现多进程并行处理的极简示例：
```python
import concurrent.futures
import os
import pathlib
from tqdm import tqdm

def process_single_wet_file(input_path: str, output_path: str):
    # 待实现：读取输入文件、处理数据、将结果写入输出路径
    return output_path

# 配置执行器（executor）
num_cpus = len(os.sched_getaffinity(0))  # 获取可用CPU核心数
executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
wet_filepaths = ["a.warc.wet.gz", "b.warc.wet.gz", "c.warc.wet.gz"]  # 示例文件路径
output_directory_path = "/path/to/output_directory/"  # 输出目录

futures = []
for wet_filepath in wet_filepaths:
    # 提取文件名并构造输出路径
    wet_filename = str(pathlib.Path(wet_filepath).name)
    output_path = os.path.join(output_directory_path, wet_filename)
    # 提交任务到执行器并获取未来对象（future）
    future = executor.submit(process_single_wet_file, wet_filepath, output_path)
    futures.append(future)

# 监控任务完成进度（带进度条）
for future in tqdm(
    concurrent.futures.as_completed(futures),
    total=len(wet_filepaths),
    desc="Processing WET files"
):
    output_file = future.result()
    print(f"输出文件已写入：{output_file}")
```

若需在 Slurm 集群上并行处理数据，可使用 `submitit` 库（提供 `concurrent.futures` 的无缝替代接口，支持自动提交任务到指定 Slurm 分区并收集结果）。以下是使用 submitit 的示例代码：
```python
import os
import pathlib
import submitit
from tqdm import tqdm

def process_single_wet_file(input_path: str, output_path: str):
    # 待实现：读取输入文件、处理数据、将结果写入输出路径
    return output_path

# 配置submitit执行器
executor = submitit.AutoExecutor(folder="slurm_logs")  # 日志存储目录
max_simultaneous_jobs = 16  # 最大并发任务数
wet_filepaths = ["a.warc.wet.gz", "b.warc.wet.gz", "c.warc.wet.gz"]  # 示例文件路径
output_directory_path = "/path/to/output_directory/"  # 输出目录

# 配置Slurm任务参数
executor.update_parameters(
    slurm_array_parallelism=max_simultaneous_jobs,
    timeout_min=15,  # 任务超时时间（分钟）
    mem_gb=2,  # 每个任务内存限制（GB）
    cpus_per_task=2,  # 每个任务CPU核心数
    slurm_account="student",  # Slurm账户名
    slurm_partition="a4-cpu",  # Slurm分区名
    slurm_qos="a4-cpu-qos"  # Slurm服务质量等级
)

futures = []
# 使用executor.batch()将任务分组为Slurm数组任务
with executor.batch():
    for wet_filepath in wet_filepaths:
        # 提取文件名并构造输出路径
        wet_filename = str(pathlib.Path(wet_filepath).name)
        output_path = os.path.join(output_directory_path, wet_filename)
        # 提交任务到执行器
        future = executor.submit(process_single_wet_file, wet_filepath, output_path)
        futures.append(future)

# 监控任务完成进度（带进度条）
for future in tqdm(
    submitit.helpers.as_completed(futures),
    total=len(wet_filepaths),
    desc="Processing WET files (Slurm)"
):
    output_file = future.result()
    print(f"输出文件已写入：{output_file}")
```
如上述代码所示，submitit 与原生 concurrent.futures API的使用方式高度相似，主要差异包括：
1. 需配置 submitit 执行器参数（指定 Slurm 任务提交目标及资源规格）；
2. 使用 executor.batch() 将所有任务分组为单个 Slurm 数组任务（而非创建与文件数相等的独立任务），以减少 Slurm 调度器负载；
3. 收集结果时使用 `submitit.helpers.as_completed() `方法。

建议使用 FastWARC 库迭代处理每个 WET 文件中的记录，使用 tldextract 库从 URL 中提取域名用于筛选。以下类可能会有所帮助：
```python
from fastwarc.warc import ArchiveIterator, WarcRecordType  # 迭代WET文件记录
from tldextract import TLDExtract  # 提取域名
```

**问题（filter_data）：6分**
(a) 编写脚本，从 Common Crawl WET 文件集合（Together 集群路径：`/data/CC/CC*.warc.wet.gz`）中筛选语言建模数据。可自由使用前序任务实现的任何筛选基元，也可探索其他筛选方法（例如基于n元语法语言模型困惑度的筛选）。核心目标是生成能使模型在 Paloma 基准测试 C4 100 domains 子集上困惑度最小化的数据。

再次强调：允许利用 Paloma 验证集构建筛选器或分类器，但严禁将验证集数据直接复制到训练数据中。脚本需统计每个筛选步骤保留的样本数量，以便明确各筛选器对最终输出数据的贡献。

交付物：
1. 并行筛选 CC WET 文件以生成语言建模数据的脚本（或脚本序列）；
2. 书面说明：每个筛选步骤剔除的样本占总剔除样本的比例。

(b) 筛选 5000 个 WET 文件需要多长时间？筛选整个 Common Crawl 数据 dump（100,000 个 WET 文件）需要多长时间？

交付物：数据筛选流水线的运行时间（含单批次 5000 个文件及全量 100,000 个文件的预估时间）。

生成语言建模数据后，我们将对其进行分析，以更好地理解数据内容。

**问题（inspect_filtered_data）：4分**
(a) 从过滤后的数据集随机选取5个示例。评论这些示例的质量，以及它们是否适合用于语言建模——尤其考虑到我们的目标是最小化C4 100领域基准测试的困惑度（perplexity）。
交付要求：过滤后数据中的5个随机示例。由于文档可能较长，仅展示相关片段即可。每个示例需附带1-2句描述，说明其是否值得用于语言建模。

(b) 选取5个被过滤脚本移除和/或修改的 CC WET 文件（Common Crawl Web Extracted Text，通用爬虫网页提取文本）。说明过滤流程中的哪个环节移除或修改了这些文档，以及你认为移除和/或修改是否合理。
交付要求：原始 WET 文件中5个随机被丢弃的示例。由于文档可能较长，仅展示相关片段即可。每个示例需附带1-2句描述，说明其移除是否合理。

(c) 如果上述分析表明需要进一步修改数据处理流程，可在训练模型前进行调整。报告你尝试过的所有数据修改和/或迭代方案。
交付要求：描述所尝试的数据修改和/或迭代方案。

在基于数据训练语言模型前，需对数据进行分词处理。使用 transformers 库中的 GPT-2 分词器，将过滤后的数据编码为整数 ID 序列以用于模型训练。切记在每个文档末尾添加 GPT-2 的序列结束标记 <<|endoftext|>。以下是示例代码：
```python
import multiprocessing
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

input_path = "path/to/your/filtered/data"  # 过滤后数据的路径
output_path = "path/to/your/tokenized/data"  # 分词后数据的输出路径

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_line_and_add_eos(line):
    return tokenizer.encode(line) + [tokenizer.eos_token_id]

with open(input_path) as f:
    lines = f.readlines()

pool = multiprocessing.Pool(multiprocessing.cpu_count())
chunksize = 100
results = []

# 多进程分词并添加序列结束标记
for result in tqdm(
    pool.imap(tokenize_line_and_add_eos, lines, chunksize=chunksize),
    total=len(lines),
    desc="Tokenizing lines"  # 分词进度提示
):
    results.append(result)

pool.close()
pool.join()

# 展平ID列表并转换为numpy数组
all_ids = [token_id for sublist in results for token_id in sublist]
print(f"已将{input_path}分词编码为{len(all_ids)}个token")
ids_array = np.array(all_ids, dtype=np.uint16)
ids_array.tofile(output_path)  # 序列化保存
```

**问题（tokenize_data）：2分**
编写脚本对过滤后的数据进行分词和序列化处理。确保按照上述示例代码的方式序列化——`ids_array.tofile(output_path)`，其中 `ids_array` 是存储整数 ID 的 `np.uint16` 类型 numpy 数组。这能保证与提供的训练脚本兼容。你的过滤后数据集包含多少个token？
交付要求：分词和序列化数据的脚本，以及生成数据集的token数量。

**模型训练**
完成数据分词后，即可基于该数据训练模型。我们将在生成的数据上训练一个 GPT-2 小型架构模型，迭代200,000步，并定期在 C4 100 领域数据集上评估验证性能。
1. 打开配置文件 `cs336-basics/configs/experiment/your_data.yaml`，将 `paths.train_bin` 属性设置为分词后训练数据的文件路径。同时需设置合适的 `training.wandb_entity` 和 `training.wandb_project` 属性以用于日志记录（Weights & Biases工具）。
2. 使用 `cs336-basics/scripts/train.py` 脚本启动训练¹。训练超参数可在 `cs336-basics/cs336_basics/train_config.py` 中查看。我们将使用 2 块 GPU 进行数据并行训练，每块 GPU 的批次大小（batch size）为128。该配置下的训练过程约需 7 小时，请合理安排时间。启动训练的命令如下：
```bash
uv run torchrun --standalone --nproc_per_node=2 scripts/train.py --config-name=experiment/your_data
```
（执行前请确保已完成上述配置文件的属性设置）

本次作业的核心目标是通过优化数据来最小化验证损失，而非通过修改模型和/或优化流程来降低损失。因此，请勿修改训练配置（上述路径和 Weights & Biases 相关属性除外）或训练脚本。

测试数据时，可将 training.save_checkpoints 配置参数设为 True，以便在每次评估验证损失时保存模型检查点（checkpoint）。设置命令如下：
```bash
uv run torchrun --standalone --nproc_per_node=2 \
scripts/train.py --config-name=experiment/your_data \
+training.save_checkpoints=True
```
模型检查点将保存至 `cs336-basics/output/your_data/step_N`（N为迭代步数）。可通过以下命令从保存的模型中生成文本样本：
```bash
uv run python scripts/generate_with_gpt2_tok.py \
--model_path cs336-basics/output/your_data/step_N
```
训练脚本地址：https://github.com/stanford-cs336/assignment4-data/blob/master/cs336-basics/scripts/train.py

**问题（train_model）：2分**
基于分词后的数据集训练语言模型（GPT-2小型架构）。定期在 C4 100 领域数据集上评估验证损失（配置文件`cs336-basics/cs336_basics/train_config.py`中默认已启用该功能）。你的模型取得的最佳验证损失是多少？将该数值提交至排行榜。
交付要求：记录的最佳验证损失值、对应的学习曲线（learning curve），以及相关操作说明。


