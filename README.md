# InEXIT Clone - Job-Resume Matching System

A deep learning-based system for matching job descriptions with resumes using BERT and Transformer architecture. This project implements a binary classification model to predict whether a candidate's resume is a "Good Fit" or "No Fit" for a given job description.

## 📋 Table of Contents
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Data Pipeline](#data-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Key Components](#key-components)

## 🏗 Architecture Overview

The system uses a sophisticated multi-stage architecture that combines:

1. **BERT Embeddings**: Pre-trained BERT model for contextual text representations
2. **Inner Interaction Layer**: Self-attention mechanism for resume and job description separately
3. **Cross Interaction Layer**: Cross-attention mechanism for resume-job matching
4. **Multi-Head Attention**: Transformer-based attention for capturing complex relationships
5. **Classification Head**: MLP layers for final matching prediction

### High-Level Flow

```
Resume Data + Job Description Data
          ↓
    BERT Tokenization
          ↓
    BERT Embeddings
          ↓
  Inner Transformers (Self-Attention)
          ↓
  Cross Transformers (Cross-Attention)
          ↓
   Feature Aggregation
          ↓
   MLP Classification
          ↓
  Match Score (0/1)
```

## 📁 Project Structure

```
inexit_clone/
├── model.py              # Neural network architecture
├── dataset.py            # Data loading and preprocessing
├── train.py              # Training loop and evaluation
├── opts.py               # Command-line arguments configuration
├── utils.py              # Utility functions (logging, metrics)
├── requirements.txt      # Python dependencies
├── InEXIT.ipynb         # Jupyter notebook for experiments
└── data/
    ├── train.csv        # Training data
    └── test.csv         # Testing data
```

## 🧠 Model Architecture

### Main Model: `BertMatchingModel`

The model is defined in [`model.py`](model.py) and consists of the following components:

#### 1. **BERT Encoder** (Lines 11-14)
```python
self.bert = AutoModel.from_pretrained(args.bert_path)
```
- Uses pre-trained BERT (default: `bert-base-uncased`)
- Output dimension: 768 (for BERT-base)
- All parameters are trainable for fine-tuning

#### 2. **Transformer Encoders** (Lines 17-20)
```python
self.encoders = nn.ModuleList([...])      # Inner interaction
self.encoders_2 = nn.ModuleList([...])    # Cross interaction
```
- **Inner Encoders**: Process resume and job separately (self-attention)
- **Cross Encoders**: Enable interaction between resume and job (cross-attention)
- Each encoder contains:
  - Multi-Head Attention layer
  - Position-wise Feed-Forward network
  - Layer Normalization
  - Residual connections

#### 3. **Classification Head** (Lines 25-29)
```python
self.mlp = nn.Sequential(
    nn.Linear(args.word_emb_dim * 3, args.hidden_size),
    nn.ReLU(),
    nn.Linear(args.hidden_size, 1)
)
```
- Input: Concatenation of [resume_vec, job_vec, resume_vec - job_vec]
- Captures both similarity and difference between resume and job
- Output: Binary classification score

### Forward Pass Flow

The forward method in [`model.py`](model.py#L45-L91) follows these steps:

1. **Metadata Embedding** (Lines 48-50): 
   - Converts 4 resume fields (summary, experience, skills, education) → `[batch, 4, 768]`
   - Converts 4 job fields (overview, responsibilities, requirements, preferred) → `[batch, 4, 768]`

2. **Text Embedding** (Lines 52-55):
   - Full resume text → `[batch, 1, 768]`
   - Full job description → `[batch, 1, 768]`

3. **Sequence Construction** (Lines 57-60):
   - Resume sequence: 5 tokens (4 metadata + 1 full text)
   - Job sequence: 5 tokens (4 metadata + 1 full text)

4. **Inner Interaction** (Lines 62-65):
   - Self-attention within resume and job independently
   - Captures internal structure and relationships

5. **Cross Interaction** (Lines 67-71):
   - Combined sequence of 10 tokens `[batch, 10, 768]`
   - Enables resume-job interaction through cross-attention

6. **Feature Aggregation** (Lines 73-79):
   - Mean pooling over sequence dimension
   - Creates resume and job vectors

7. **Classification** (Lines 81-86):
   - Concatenates: `[resume_vec, job_vec, abs_difference]`
   - Predicts match score via MLP

### Attention Mechanism

The [`Scaled_Dot_Product_Attention`](model.py#L89-L103) class implements:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The [`Multi_Head_Attention`](model.py#L120-L159) splits attention into multiple heads:
- Number of heads: 8 (configurable)
- Each head dimension: 768 / 8 = 96
- Enables capturing different types of relationships

## 📊 Data Pipeline

### Dataset: `PJFDataset`

Defined in [`dataset.py`](dataset.py), the dataset class handles:

#### Input Format (CSV)
The data files should contain the following columns:
- **Resume fields**: `resume_summary`, `resume_experience`, `resume_skills`, `resume_education`, `resume_text`
- **Job fields**: `jd_overview`, `jd_responsibilities`, `jd_requirements`, `jd_preferred`, `job_description_text`
- **Label**: `label` ("Good Fit" or "No Fit")

#### Data Processing Pipeline (Lines 26-71)

1. **Metadata Construction** (Lines 30-43):
   ```python
   resume_metadata = [
       "Summary: " + str(row['resume_summary']),
       "Experience: " + str(row['resume_experience']),
       "Skills: " + str(row['resume_skills']),
       "Education: " + str(row['resume_education'])
   ]
   ```

2. **Tokenization** (Lines 45-63):
   - Metadata fields: max length 64 tokens (configurable via `max_feat_len`)
   - Full text: max length 256 tokens (configurable via `max_sent_len`)
   - Padding to max length for batch processing

3. **Output Format** (Lines 74-84):
   - 8 tensors for input (4 pairs of input_ids + attention_masks)
   - 1 tensor for label (0.0 or 1.0)

### Data Split Strategy

In [`train.py`](train.py#L33-L42), the training data is split with stratification:
```python
train_df, valid_df = train_test_split(
    df_all, 
    test_size=0.2, 
    random_state=args.seed, 
    stratify=df_all['label']  # Maintains class balance
)
```

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd inexit_clone
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

This will install:
- `transformers==4.36.2` - Hugging Face transformers for BERT
- `torch>=2.0.0` - PyTorch deep learning framework
- `pandas==2.1.4` - Data manipulation
- `scikit-learn==1.3.2` - Metrics and data splitting
- Other utilities (tqdm, numpy)

3. **Download BERT model** (optional):
```bash
# Model will auto-download on first run
# Or pre-download:
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
```

## 💻 Usage

### Training

Run the training script from [`train.py`](train.py):

```bash
python train.py \
  -data_path ./data \
  -bert_path bert-base-uncased \
  -gpu 0 \
  -train_batch_size 16 \
  -num_epochs 10 \
  -learning_rate 2e-5 \
  -seed 42
```

### Training Process

The training loop in [`train.py`](train.py#L60-L101) includes:

1. **Model Training** (Lines 60-81):
   - Forward pass through model
   - BCE with Logits loss calculation
   - Backward propagation
   - Gradient clipping (max norm: 1.0)
   - AdamW optimization

2. **Validation** (Lines 83-95):
   - Evaluation on validation set
   - Metrics computation (AUC, Accuracy, F1)
   - Best model checkpoint saving

3. **Early Stopping** (Lines 103-105):
   - Stops if no improvement after 3 epochs (configurable)

4. **Testing** (Lines 107-122):
   - Loads best checkpoint
   - Evaluates on test set
   - Reports final metrics

### Metrics Computed

The [`compute_metrics`](train.py#L55-L66) function calculates:
- **AUC**: Area Under ROC Curve
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 * P * R / (P + R)

### Using Jupyter Notebook

For interactive experimentation, use [`InEXIT.ipynb`](InEXIT.ipynb):
```bash
jupyter notebook InEXIT.ipynb
```

Or run in Google Colab:
1. Mount Google Drive
2. Navigate to project directory
3. Install requirements
4. Run training command

## ⚙ Configuration

### Command-Line Arguments

All arguments are defined in [`opts.py`](opts.py):

#### General Setup (Lines 11-22)
| Argument | Default | Description |
|----------|---------|-------------|
| `-token` | "BERT-Match" | Experiment identifier |
| `-data_path` | "data" | Data directory path |
| `-bert_path` | "bert-base-uncased" | BERT model path |
| `-save_path` | "save" | Checkpoint save directory |
| `-log_dir` | "log" | Log file directory |
| `-gpu` | 0 | GPU device ID |

#### Training Hyperparameters (Lines 24-49)
| Argument | Default | Description |
|----------|---------|-------------|
| `-train_batch_size` | 16 | Training batch size |
| `-valid_batch_size` | 16 | Validation batch size |
| `-test_batch_size` | 16 | Test batch size |
| `-learning_rate` | 2e-5 | Learning rate (typical for BERT) |
| `-weight_decay` | 1e-4 | L2 regularization |
| `-max_gradient_norm` | 1.0 | Gradient clipping threshold |
| `-num_epochs` | 10 | Maximum training epochs |
| `-end_step` | 3 | Early stopping patience |

#### Model Architecture (Lines 31-43)
| Argument | Default | Description |
|----------|---------|-------------|
| `-max_feat_len` | 64 | Max tokens for metadata fields |
| `-max_sent_len` | 256 | Max tokens for full text |
| `-word_emb_dim` | 768 | BERT embedding dimension |
| `-hidden_size` | 768 | Hidden layer size |
| `-num_heads` | 8 | Attention heads |
| `-num_layers` | 1 | Number of transformer layers |
| `-dropout` | 0.1 | Dropout rate |

## 🔧 Key Components

### 1. Utility Functions ([`utils.py`](utils.py))

- **`parse_args()`** (Lines 9-17): Parses command-line arguments
- **`get_logger()`** (Lines 31-47): Initializes logging system
- **`get_parameter_number()`** (Lines 19-23): Counts model parameters
- **`classify()`** (Lines 49-63): Computes confusion matrix for metrics
- **`keep_only_the_best()`** (Lines 25-29): Saves best model checkpoint

### 2. Model Components ([`model.py`](model.py))

- **`BertMatchingModel`** (Lines 7-91): Main matching model
- **`Encoder`** (Lines 106-113): Transformer encoder block
- **`Multi_Head_Attention`** (Lines 115-159): Multi-head self-attention
- **`Position_wise_Feed_Forward`** (Lines 162-174): FFN with residual connection
- **`Scaled_Dot_Product_Attention`** (Lines 89-103): Scaled dot-product attention

### 3. Dataset Handler ([`dataset.py`](dataset.py))

- **`PJFDataset`** (Lines 5-86): Custom PyTorch dataset
  - Handles CSV loading
  - BERT tokenization
  - Label mapping
  - Batch preparation

## 📈 Training Tips

1. **Learning Rate**: Start with 2e-5 for BERT fine-tuning (typical range: 2e-5 to 5e-5)
2. **Batch Size**: Adjust based on GPU memory (16 works well for most GPUs)
3. **Sequence Length**: 
   - Shorter for metadata (64 tokens)
   - Longer for full text (256 tokens)
4. **Early Stopping**: Prevents overfitting by stopping after 3 epochs without improvement
5. **Gradient Clipping**: Helps stabilize training (max norm: 1.0)

## 📊 Expected Performance

The model should achieve:
- **AUC**: ~0.85-0.95 (depending on data quality)
- **Accuracy**: ~0.80-0.90
- **F1 Score**: ~0.75-0.90

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce batch size: `-train_batch_size 8`
- Reduce sequence length: `-max_sent_len 128`
- Use gradient accumulation

### BERT Download Issues
- Pre-download model manually
- Use local path: `-bert_path /path/to/bert`

### Poor Performance
- Increase training epochs: `-num_epochs 20`
- Adjust learning rate: `-learning_rate 3e-5`
- Check data quality and class balance

## 📝 Citation

If you use this codebase, please cite:
```bibtex
@misc{inexit_clone,
  title={InEXIT Clone: Job-Resume Matching System},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/lightningflashh/inexit_clone}}
}
```

## 📄 License

This project is available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

---

**Note**: This is a research/educational project. Ensure proper data privacy and ethical considerations when using with real resume and job description data.
