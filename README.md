# Computer Vision Specialist Track

**An 8-Week Full-Time Training Plan for Experienced Engineers**

| Attribute | Detail |
|-----------|--------|
| Audience | Software engineers experienced on other stacks, new to CV / ML |
| Format | Full-time, 32–40 hours/week · self-directed · minimal mentorship |
| Structure | 5 phases across 8 weeks, mapped to four CV capability pillars |

---

## Program at a Glance

| Week | Pillar | Focus | Success Gate |
|------|--------|-------|-------------|
| 1 | Foundations | ML & PyTorch fundamentals, the training loop, image basics | CNN ≥70% on CIFAR-10 + written explainer |
| 2 | Vision Intelligence Core | Classification, transfer learning, augmentation | Fine-tuned model beats baseline; ablation table |
| 3 | Vision Intelligence Core | Detection, segmentation, tracking, anomaly detection | YOLO detector at target mAP + tracked video |
| 4 | Document Intelligence | OCR + managed document-understanding services | KV+table→JSON via self-hosted & managed; comparison matrix |
| 5 | Vision + Generative AI | VLMs, VQA, NL explanation, captioning & multimodal embeddings | VLM schema-valid answers + caption/embed retrieval works |
| 6 | Vision + Generative AI | Multimodal RAG: layout parsing, indexing, multimodal retrieval | Text & text+image queries retrieve correct grounded context |
| 7 | Vision + Generative AI | Agentic multimodal RAG: planning, tools, self-correction | Agent passes eval suite; self-corrects on a failure case |
| 8 | Industrial-Grade Deployment | Optimize/serve/monitor the Wk-2 classifier + serve the RAG system | Served API, drift experiment flags injected drift, Grad-CAM |

---

## Repository Structure

```
ComputerVision/
├── README.md                  # This file
├── requirements.txt           # Global / shared dependencies
├── .gitignore
├── setup_env.sh               # One-time environment setup script (WSL2)
├── setup_env.ps1              # One-time environment setup script (PowerShell)
│
├── week1_foundations/         # ML & CV Foundations
├── week2_classification/      # Image Classification & Transfer Learning
├── week3_detection/           # Detection, Segmentation, Tracking, Anomaly
├── week4_documents/           # Document Understanding
├── week5_vlms/                # VLMs, VQA, Captioning & Multimodal Embeddings
├── week6_multimodal_rag/      # Multimodal RAG Pipeline
├── week7_agentic_rag/         # Agentic Multimodal RAG
└── week8_deployment/          # ML Model Deployment & Tracking
```

Each week folder contains its own `README.md` with day-by-day instructions, deliverables, and results.

---

## Environment Setup

### Prerequisites
- Python 3.12+
- NVIDIA GPU (recommended) or Google Colab / Kaggle Notebooks (free T4)
- Ubuntu 24.04 on WSL2 (for local work)
- Git

### Quick Start

```bash
# Clone the repo
git clone <your-repo-url>
cd ComputerVision

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/WSL2
# OR
.\.venv\Scripts\Activate   # Windows PowerShell

# Install shared dependencies
pip install -r requirements.txt

# Each week has its own requirements
pip install -r week1_foundations/requirements.txt
```

### Experiment Tracking
- **Weights & Biases**: `wandb login` (free tier)
- **MLflow**: Used from Week 8 onward

---

## Daily Rhythm (Suggested)

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | ~2 hrs | Guided learning (video / reading) |
| Afternoon | ~4–5 hrs | Hands-on building & coding |
| Evening | ~1 hr | Writing notes, logging results, committing to Git |

> End each day with a working commit. End each week by shipping the deliverable.

---

## Resources (All Weeks)

**When stuck:**
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Hugging Face Forums & Discord](https://discuss.huggingface.co/)
- [r/computervision](https://www.reddit.com/r/computervision/)
- [Stack Overflow — pytorch / computer-vision](https://stackoverflow.com/questions/tagged/pytorch+computer-vision)

**Reference & depth:**
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [Papers with Code](https://paperswithcode.com/)
- [Roboflow Blog](https://blog.roboflow.com/)
- [Hugging Face — Computer Vision Course](https://huggingface.co/learn/computer-vision-course)
