# 🧠 AI Learning Psychology Analyzer

AI-powered system that analyzes student attention, cognitive load, and learning styles in real-time.

## 🎯 Features

- **Attention Tracking**: Real-time attention level monitoring (67.8% accuracy)
- **Cognitive Load Assessment**: Mental workload analysis (37.6% accuracy)
- **Learning Style Detection**: Personalized learning preferences (21.3% accuracy)
- **Adaptive Recommendations**: Dynamic content adjustments

## 🚀 Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run demo**:
```bash
python demo.py
```

3. **Train models** (optional):
```bash
python train_models_real_data.py
```

## 📊 Usage

```python
from src.ai.core.psychology_analyzer import LearningPsychologyAnalyzer

analyzer = LearningPsychologyAnalyzer()
analysis = analyzer.analyze_learning_session(data)

print(f"Attention Level: {analysis['attention_level']}")
print(f"Cognitive Load: {analysis['cognitive_load']}")
print(f"Learning Style: {analysis['learning_style']}")
```

## 🔧 Technical Stack

- **Python 3.11+**
- **scikit-learn, pandas, numpy**
- **Real educational data from 2,300+ samples**
- **Clean Architecture with SOLID principles**

## 📁 Project Structure

```
src/
├── ai/
│   ├── core/           # Main psychology analyzer
│   ├── models/         # AI models (attention, cognitive load, learning style)
│   └── data/           # Data preprocessing
├── models/             # Trained model files
├── data/               # Educational datasets
└── demo.py             # Demo application
```

## 📄 License

MIT License - see LICENSE file for details. 