# CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming

[![arXiv](https://img.shields.io/badge/arXiv-2505.12925-b31b1b.svg)](https://arxiv.org/abs/2505.12925)
[![🤗 HF Collection](https://img.shields.io/badge/HuggingFace-CPRet-yellow)](https://huggingface.co/collections/coldchair16/cpret-682451276f05c5988fcbdf34)

Email contact: 2317757009@qq.com

## 🌐 Try Online Demo

We provide an **online demo** of the CPRet retrieval service, available at:

👉 [https://www.cpret.online/](https://www.cpret.online/)

The former IP address, http://1.94.255.218:5000/, should be updated to the new domain.

This demo can assist in **duplicate problem detection** by retrieving potentially similar problems, though final identification still requires manual verification.

It also supports **similar problem retrieval** to help broaden your problem-solving perspective.

You can input either a **full problem description** or a **simplified version**, and the system will return the most relevant existing problems.

You can refer to the usage examples of the retrieval platform at: [https://github.com/coldchair/CPRet/blob/main/TestCases.md](https://github.com/coldchair/CPRet/blob/main/TestCases.md)

It runs the same codebase and embedding model as the local deployment (see below), so you can preview its capabilities before setting up your own instance.

## 🚀 News

**July 2025: CPRetriever-Prob-Qwen3-4B Released with Enhanced Retrieval Performance!**

We're excited to announce a major update to the CPRetriever model series! We've trained the new [**CPRetriever-Prob-Qwen3-4B**](https://huggingface.co/coldchair16/CPRetriever-Prob-Qwen3-4B) model based on [**Qwen3-Embedding-4B**](https://huggingface.co/Qwen/Qwen3-Embedding-4B), released in June 2025, and it has achieved **state-of-the-art results** in problem-related retrieval tasks. Concurrently, we've also updated our website's retrieval problem database to the latest July 2025 version.

Here's a comparison of model performance:

| model | type | size | Text-to-Code | Code-to-Code | Problem-to-Duplicate | Simplified-to-Full | Avg |
| :------------------------ | :--- | :--- | :----------- | :----------- | :------------------- | :----------------- | :----- |
| CPRetreiver-Code | code | 2B | 70.40 | 70.59 | 38.68 | 81.45 | 65.28 |
| CPRetreiver-Prob | code | 2B | 56.50 | 70.68 | 60.06 | 90.74 | 69.50 |
| CPRetriever-Prob-Qwen3-4B | code | 4B | 65.85 | 70.19 | 71.45 | 95.03 | 75.63 |

## 📌 Overview

**CPRet** is a comprehensive suite for competitive programming retrieval research, consisting of:

* A large-scale dataset and benchmark for retrieval tasks in coding contests.
* A dual-stage training pipeline with contrastive pretraining and task-specific fine-tuning.
* A local retrieval server for **simplified description** and **duplicate problem** search, powered by our trained model **[CPRet-Prob](https://huggingface.co/coldchair16/CPRetriever-Prob)** (based on [Salesforce/SFR-Embedding-Code-2B\_R](https://huggingface.co/Salesforce/SFR-Embedding-Code-2B_R)).

We define the following **four core retrieval tasks** to support both practical applications and academic benchmarking:

1. **Text-to-Code (T2C):** Retrieve relevant code given a natural language problem description.
2. **Code-to-Code (C2C):** Retrieve other implementations of the same problem based on a given solution.
3. **Problem-to-Duplicate (P2D):** Detect duplicate or near-duplicate problems from existing contest archives.
4. **Simplified-to-Full (S2F):** Retrieve the original full version of a simplified problem.


## 🧰 Repository Contents

* `cp-retrieval-server/`: Code for running a local retrieval web service.
* `stage1/`: Code for stage-1 contrastive pretraining.
* `stage2/`: Code for stage-2 problem-level fine-tuning.

---

## ⚙️ Setup

### Environment

* Recommended: `python >= 3.10`

* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

* Install PyTorch (with CUDA support if needed):
  → Refer to: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

* PyTorch ≥ 2.0 is recommended.

### 🔁 Accessing Hugging Face from Restricted Regions

If you're experiencing connectivity issues with Hugging Face, consider using the official mirror:

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

Or set it as an environment variable:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 🚀 Run Local Retrieval Service


1.  **Download embeddings:**

    * **If you are using the new model, `CPRetriever-Prob-Qwen3-4B`:**
        * Please download the following files from [HF dataset CPRet-Embeddings](https://huggingface.co/datasets/coldchair16/CPRet-Embeddings) into the `cp-retrieval-server/` directory:
            * `probs_2507.jsonl`
            * `probs_2507_embs.npy`

    * **If you are using the old model, `CPRetriever-Prob`:**
        * Please download the following files from [HF dataset CPRet-Embeddings](https://huggingface.co/datasets/coldchair16/CPRet-Embeddings) into the `cp-retrieval-server/` directory:
            * `probs_embs.npy`
            * `probs.jsonl`




2.  **Start the service:**

    ```bash
    cd cp-retrieval-server
    ```

    If you're using the **old model (`CPRetriever-Prob`)**, set the following environment variables before starting the service:

    ```bash
    export MODEL_PATH=coldchair16/CPRetriever-Prob
    export EMB_PATH=./probs_embs.npy
    export PROB_PATH=./probs.jsonl
    ```

    Then, run:

    ```bash
    python app.py
    ```

3. **About the Dataset:**

   The current dataset includes problems from the following online judges:

   * [Codeforces](https://codeforces.com/)
   * [AtCoder](https://atcoder.jp/)
   * [SPOJ](https://www.spoj.com/)
   * [Nowcoder](https://ac.nowcoder.com/)
   * [Luogu](https://www.luogu.com.cn/)
   * [Loj](https://loj.ac/)

   The data is collected up to **July 2025**.
   You can add your own data source and generate embeddings using [`compute_embs.py`](cp-retrieval-server/compute_embs.py). Running this process for the current database on an A800 GPU takes approximately 4.5 GPU hours.

   If you have access to a larger or more diverse problem dataset, **we welcome contributions and are happy to update the collection** — feel free to contact us (231775009@qq.com) or open an issue/pull request.


4.  **System Requirements:**

    This service can be run on **CPU** or **GPU**, depending on your environment.
    We recommend the following memory for smooth operation:

    * For the **2B old models** (e.g., `CPRetriever-Prob`): at least **16GB of system memory or GPU VRAM**.
    * For the **4B new model** (`CPRetriever-Prob-Qwen3-4B`): **32GB or more of system memory or GPU VRAM**.

    Typical query latency:

    * On **CPU** (8 cores): **10–20 seconds**.
    * On **GPU** (e.g., A800): **0.1–1 seconds**.


---

## 🏋️‍♀️ Training Instructions

> **⚠️ Note**: Recommended GPU memory ≥ **50 GB** to avoid OOM.

## 🔧 Stage 1: Contrastive Pretraining

```bash
cd stage1
torchrun --nproc_per_node=8 train.py
````

* Change `--nproc_per_node` to match the number of available GPUs.
* Use `--help` to see all configurable hyperparameters.

### ⚠️ Note on Using `Salesforce/SFR-Embedding-Code-2B_R`

If you are using [`Salesforce/SFR-Embedding-Code-2B_R`](https://huggingface.co/Salesforce/SFR-Embedding-Code-2B_R) as your encoder, make sure to **manually disable `device_map="auto"`** when loading the model.

The original code might look like this:

```python
self.model = Gemma2Model.from_pretrained(config._name_or_path, trust_remote_code=True, is_causal=False, device_map="auto")
self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path, trust_remote_code=True, device_map="auto")
```

This setting can cause the model to skip training due to automatic device placement.
**Please change it to:**

```python
self.model = Gemma2Model.from_pretrained(config._name_or_path, trust_remote_code=True, is_causal=False, device_map=None)
self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path, trust_remote_code=True, device_map=None)
```

Alternatively, you can directly copy the patched file from our repo:
👉 [modeling\_gemma2.py](https://huggingface.co/coldchair16/CPRetriever-Code/blob/main/modeling_gemma2.py)


---
### Stage 2: Problem-Level Fine-Tuning

```bash
cd stage2
torchrun --nproc_per_node=1 train.py
```

* Also supports `--help` to inspect all args.

---

## 🔧 Notable Hyperparameters

* `--model_path`: Can be either an HF model repo (e.g. `coldchair16/CPRetriever-Code`) or a local directory supporting SentenceTransformer.
* `--eval_only True`: Run evaluation without training.


## 📫 Citation & License

If you find **CPRet** useful in your research or applications, please consider citing our paper:

```bibtex
@misc{deng2025cpretdatasetbenchmarkmodel,
  title     = {CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming},
  author    = {Han Deng and Yuan Meng and Shixiang Tang and Wanli Ouyang and Xinzhu Ma},
  year      = {2025},
  eprint    = {2505.12925},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url       = {https://arxiv.org/abs/2505.12925}
}
```

### 📄 License

This work is released for **research and non-commercial use only**.

> **License**: CC BY-NC 4.0 (Attribution-NonCommercial)
> [📜 View full license text](https://creativecommons.org/licenses/by-nc/4.0/)


