# CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming

**[Hugging Face Collection](https://huggingface.co/collections/coldchair16/cpret-682451276f05c5988fcbdf34)**
**[arXiv Paper (coming soon)](#)**

## 📌 Overview

**CPRet** is a comprehensive suite for competitive programming retrieval research, consisting of:

* A large-scale dataset and benchmark for retrieval tasks in coding contests.
* A dual-stage training pipeline with contrastive pretraining and task-specific fine-tuning.
* A local retrieval server powered by sentence-transformer-based models.

We target four core retrieval tasks in the competitive programming domain, enabling both practical applications (e.g., search, deduplication) and academic benchmarking.

---

## 🌐 Try Online Demo

We provide an **online demo** of the CPRet retrieval service, available at:

👉 [http://1.94.255.218:5000/](http://1.94.255.218:5000/)

This demo is powered by the same codebase and embedding model as the local deployment instructions below. Feel free to try it out before setting up your own instance!

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

1. **Download embeddings:**

   * From: [HF dataset CPRet-Embeddings](https://huggingface.co/datasets/coldchair16/CPRet-Embeddings)
   * Download the following files into `cp-retrieval-server/`:

     * `probs_embs.npy`
     * `probs.jsonl`

2. **Start the service:**

   ```bash
   cd cp-retrieval-server
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

   The data is collected up to **May 2025**.
   You can add your own data source and generate embeddings using [`compute_embs.py`](cp-retrieval-server/compute_embs.py).
   If you have access to a larger or more diverse problem dataset, **we welcome contributions and are happy to update the collection** — feel free to [contact us](231775009@qq.com) or open an issue/pull request.

4. **System Requirements:**

   This service can be run on **CPU** or **GPU**, depending on your environment.
   We recommend at least **16GB of system memory or GPU VRAM** for smooth operation.

   * On **CPU**: With 8 cores, typical query latency is **10–20 seconds**.
   * On **GPU**: With an A800, most queries respond in **0.1–1 seconds**.

---

## 🏋️‍♀️ Training Instructions

> **⚠️ Note**: Recommended GPU memory ≥ **50 GB** to avoid OOM.

### Stage 1: Contrastive Pretraining

```bash
cd stage1
torchrun --nproc_per_node=8 train.py
```

* Change `--nproc_per_node` to match your available GPUs.
* Use `--help` to see full hyperparameter options.

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

---

## 📫 Citation & License

* Citation information and license will be released along with the paper.
