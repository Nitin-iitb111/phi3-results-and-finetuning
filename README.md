# **📘 Phi-3 Vision: Domain-Specific Multimodal Reasoning & Finetuning**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-important)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)

---

## **📄 Abstract**

This project evaluates and extends **Phi-3 Vision**, a compact multimodal (image–text) model, for domain-specific reasoning tasks such as captioning, OCR-style interpretation, and visual question answering. We benchmark the base model under academic hardware constraints and investigate three efficient adaptation strategies: **prompt engineering**, **LoRA fine-tuning**, and **feature-level adapters**.
Experiments conducted on a Tesla T4 GPU reveal that the base model struggles with out-of-domain tasks; however, parameter-efficient techniques significantly improve model specialization while remaining cost-effective.

---

## **🧠 Model Overview**

**Phi-3 Vision**, released by Microsoft, is a small multimodal model designed for efficient deployment and finetuning.

Key features:

* Lightweight architecture suitable for 16GB GPUs
* Unified vision–language transformer pipeline
* Strong zero-shot generalization relative to its size
* Ideal for research, prototyping, and finetuning on compute-limited systems

---

## **⚙️ Methodology**

### **1. Prompt Engineering**

* Handcrafted instructional prompts
* Few-shot demonstrations
* Zero-cost baseline for task adaptation

### **2. LoRA Fine-Tuning**

* Low-Rank Adaptation applied to attention projections
* Minimal trainable parameters
* Works efficiently on a **single T4 GPU**
* Equation:
  [
  W' = W + BA
  ]
  where ( r \ll d )

### **3. Feature-Level Adapters (CLIP-Adapter Inspired)**

* Small MLP/linear adapters added to frozen vision embeddings
* Helps reduce distribution shift for OCR, diagrams, scientific content
* Lightweight & fast to train

---

## **🧪 Experiments & Datasets**

Evaluations were performed on:

* **COCO-style captioning samples**
* **ScienceQA** (multimodal QA)
* **Text-in-Image datasets**, including OCR-heavy tasks

Metrics computed:

* BLEU-1/4
* ROUGE-1/2/L
* Exact Match
* Numeric accuracy
* Inference latency

All experiments used:

* **GPU:** Tesla T4 (15.8GB)
* **Frameworks:** PyTorch, HuggingFace Transformers, PEFT

---

## **📁 Folder Structure**

```
├── phi3_finetune.ipynb              # LoRA & adapter training
├── Phi3_result_and_finetune.ipynb   # Evaluation, metrics & plots
├── Report_and_resuults/             # Reports
├── README.md                        # This file
```

---

## **🚀 How to Run the Code**

### **1. Clone Repository**

```bash
git clone https://github.com/Nitin-iitb111/phi3-results-and-finetuning
cd phi3-results-and-finetuning
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run Jupyter/Colab Notebooks**

* `phi3_finetune.ipynb` → LoRA & Feature Adapter training
* `Phi3_result_and_finetune.ipynb` → Evaluation & visualizations

### **4. Basic Inference Example**

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-vision")
processor = AutoProcessor.from_pretrained("microsoft/Phi-3-vision")

inputs = processor(image, prompt, return_tensors="pt")
outputs = model.generate(**inputs)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

---

## **🔧 Dependencies & Environment**

### **Core Packages**

* Python ≥ 3.9
* PyTorch ≥ 2.1
* Transformers ≥ 4.40
* PEFT
* Accelerate
* BitsAndBytes (for QLoRA)
* Pillow
* torchvision

### **Hardware Requirement**

* **1× Tesla T4 (16GB)** or equivalent

---

## **🔮 Future Work**

* Larger-scale finetuning on full ScienceQA
* Ablation studies on LoRA rank, adapter depth, prompt length
* More robust OCR adapter for varied text-in-image data
* Benchmarking on chart-VQA, diagram reasoning, and handwritten text
* Optimization for faster inference on CPU/GPU

---

## **👥 Authors**

**Nitin Yadav (22b3957)**
Dual Degree (EE), IIT Bombay
Email: *[22b3957@iitb.ac.in](mailto:22b3957@iitb.ac.in)*

**Anisha Saini (22b3943)**
Dual Degree (EE), IIT Bombay
Email: *[22b3943@iitb.ac.in](mailto:22b3943@iitb.ac.in)*

---
