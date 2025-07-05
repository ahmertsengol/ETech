---
title: AI Learning Psychology Analyzer
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
license: mit
tags:
- education
- psychology
- learning-analytics
- attention-tracking
- cognitive-load
- learning-styles
- sklearn
- random-forest
- educational-ai
---

# AI Learning Psychology Analyzer

Three specialized machine learning models for educational psychology analysis, trained on real educational datasets.

## Models

| Model | Accuracy | Purpose |
|-------|----------|---------|
| **Attention Tracker** | 67.8% | Monitor student attention levels |
| **Cognitive Load Assessor** | 37.6% | Evaluate cognitive burden |
| **Learning Style Detector** | 21.3% | Identify learning preferences |

## Quick Start

```bash
pip install huggingface-hub scikit-learn pandas numpy
```

```python
from huggingface_hub import hf_hub_download
import pickle

# Download and load model
model_path = hf_hub_download(
    repo_id="Lazzaran/ai-learning-psychology-analyzer",
    filename="attention_tracker_model.pkl"
)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Make prediction
prediction = model.predict([[0.8, 0.7, 0.9, 0.6, 0.8]])
```

## Model Details

### Attention Tracker
- **Type**: Random Forest Classifier
- **Features**: 5 educational engagement metrics
- **Classes**: Critical, Low, Medium, High
- **Training**: 1,840 samples

### Cognitive Load Assessor
- **Type**: Random Forest Classifier  
- **Features**: 5 cognitive load indicators
- **Classes**: Optimal, Moderate, Overloaded, Critical
- **Training**: 1,840 samples

### Learning Style Detector
- **Type**: Random Forest Classifier
- **Features**: 5 learning behavior metrics
- **Classes**: Visual, Auditory, Reading, Kinesthetic, Multimodal
- **Training**: 1,840 samples

## Technical Specifications

- **Framework**: Scikit-learn
- **Preprocessing**: StandardScaler normalization
- **Validation**: Stratified K-fold cross-validation
- **Data**: Real educational datasets (2,300 records)
- **Regularization**: Applied to prevent overfitting

## Use Cases

- **Personalized Learning**: Adapt content based on student psychology
- **Early Intervention**: Identify struggling students
- **Learning Optimization**: Match teaching methods to learning styles
- **Educational Research**: Study attention patterns and cognitive load

## Citation

```bibtex
@software{ai_learning_psychology_analyzer,
  title={AI Learning Psychology Analyzer},
  author={Lazzaran},
  year={2024},
  url={https://huggingface.co/Lazzaran/ai-learning-psychology-analyzer}
}
```

## License

MIT 