# Qwen Fine-Tuning & Reasoning Pipeline 🚀

A professional, portfolio-ready repository for fine-tuning Large Language Models (LLMs) using **Qwen-2.5**, **LoRA/QLoRA**, and Hugging Face tools in a hybrid **VS Code + Google Colab** workflow.

---

## 📅 Learning Curriculum (10 Lessons)

- **Lesson 1: Environment Setup** 🛠️
  - Local workspace setup in VS Code, Git initialization, and configuration decoupling.
- **Lesson 2: Explore Dataset** 📊
  - Discovering dataset templates, cleansing conversations, and analyzing sequence lengths.
- **Lesson 3: Tokenization** 🔠
  - Understanding Vocabs, BPE Tokenizers, Padding, Truncation, and Attention Masks.
- **Lesson 4: Understanding Qwen** 🧠
  - Diving into the Qwen-2.5 architecture, attention structures, and model parameters.
- **Lesson 5: LoRA (Low-Rank Adaptation)** 📐
  - Math behind adapter matrices, frozen weights, and setting target projection layers.
- **Lesson 6: QLoRA (Quantized LoRA)** 📉
  - Loading models in 4-bit precision using `bitsandbytes`, double quantization, and NF4 type.
- **Lesson 7: Fine-Tuning Execution** ⚡
  - Running supervised fine-tuning (`SFTTrainer`) with gradient accumulation and cosine scheduling.
- **Lesson 8: Evaluation & Loss Curves** 📈
  - Validating performance, inspecting training vs. validation loss, and computing perplexity.
- **Lesson 9: Merging & Deployment** ⚙️
  - Merging LoRA adapters back to base models and exporting them to GGUF format.
- **Lesson 10: Build Your Own ChatGPT** 💬
  - Integrating our fine-tuned Qwen model with **Ollama** and **Open WebUI** for local usage.

---

## 📁 Repository Structure

```
qwen-finetuning/
│
├── data/
│   ├── raw/           # Unprocessed source datasets
│   ├── processed/     # Tokenized or formatted chat sequences
│   └── cache/         # Local dataset cache
│
├── notebooks/         # Jupyter Notebooks for exploration and lessons
│   ├── 01_dataset_exploration.ipynb
│   └── 02_tokenization.ipynb
│
├── src/               # Reusable Python scripts
│   ├── dataset.py     # Data parsing and formatting
│   ├── tokenizer.py   # Tokenization wrappers
│   ├── train.py       # Fine-tuning entrypoint
│   ├── evaluate.py    # Metric and loss evaluation
│   ├── inference.py   # Interactive query/response script
│   └── utils.py       # Logging and configuration loaders
│
├── configs/
│   └── qwen.yaml      # decoupled model, LoRA & training hyperparameters
│
├── outputs/           # Training checkpoints
├── adapters/          # Fine-tuned adapter weights
├── merged/            # Merged base + adapter models
├── requirements.txt   # Package dependencies
├── index.html         # Interactive LLM Fine-Tuning Guide
└── README.md          # Project instructions
```

---

## 🔄 VS Code + Google Colab Workflow

Since training LLMs requires high-performance GPUs (which are not available on standard local CPUs), we adopt the following hybrid development strategy:

1. **Write locally in VS Code**: Edit python scripts, define configurations in `qwen.yaml`, and organize code.
2. **Push changes to GitHub**: 
   ```bash
   git add .
   git commit -m "update training configuration"
   git push
   ```
3. **Open Google Colab**: Spin up a notebook, select a T4/A100 GPU runtime, clone the repository, install dependencies, and run training.
   ```python
   # Inside a Google Colab Cell:
   !git clone https://github.com/your-username/qwen-finetuning.git
   %cd qwen-finetuning
   !pip install -r requirements.txt
   !python src/train.py
   ```
4. **Push Adapters**: Save the checkpoints and adapters to Hugging Face Hub or download them directly from Colab's workspace.

---

## 📖 Get Started

Open the interactive **[LLM Fine-Tuning Guide](file:///c:/Srinivas/FinetuningLLM/index.html)** in your browser to inspect visual box diagrams explaining Tokenization, LoRA weight math, and training metrics!
