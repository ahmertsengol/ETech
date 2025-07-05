# 🚀 Hugging Face'e Model Yükleme Rehberi

Bu rehber, AI Learning Psychology Analyzer modellerinizi Hugging Face Hub'a yüklemeniz için gerekli tüm adımları içerir.

## 📋 Gereksinimler

### 1. Hugging Face Hesabı
- [Hugging Face](https://huggingface.co/) sitesine kaydolun
- Hesabınızı doğrulayın

### 2. API Token
- [Hugging Face Settings](https://huggingface.co/settings/tokens) sayfasına gidin
- "New token" butonuna tıklayın
- Token adı verin (örn: "AI-Learning-Psychology-Models")
- "Write" yetkisi verin
- Token'ı kopyalayın ve güvenli bir yere saklayın

### 3. Gerekli Paketler
```bash
pip install huggingface-hub>=0.19.0
```

## 🔧 Kurulum Adımları

### 1. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Modellerin Hazır Olduğunu Kontrol Edin
```bash
ls -la models/
```

Şu dosyaları görmelisiniz:
- `attention_tracker_model.pkl`
- `attention_tracker_scaler.pkl`
- `attention_tracker_encoder.pkl`
- `cognitive_load_assessor_model.pkl`
- `cognitive_load_assessor_scaler.pkl`
- `learning_style_detector_model.pkl`
- `learning_style_detector_scaler.pkl`
- `learning_style_detector_encoder.pkl`
- `training_metrics.pkl`
- `training_report.txt`

### 3. Environment Variables Ayarlayın (Opsiyonel)
```bash
export HUGGINGFACE_USERNAME="your-username"
export HUGGINGFACE_TOKEN="your-token-here"
```

## 🚀 Upload İşlemi

### Yöntem 1: Komut Satırı Argumentları
```bash
python upload_to_huggingface.py --username your-username --token your-token
```

### Yöntem 2: Environment Variables
```bash
# Önce environment variables ayarlayın
export HUGGINGFACE_USERNAME="your-username"
export HUGGINGFACE_TOKEN="your-token-here"

# Sonra script'i çalıştırın
python upload_to_huggingface.py
```

### Yöntem 3: Özel Repository Adı
```bash
python upload_to_huggingface.py --username your-username --token your-token --repo custom-model-name
```

### Dry Run (Test Modu)
```bash
python upload_to_huggingface.py --username your-username --token your-token --dry-run
```

## 📊 Upload Sürecinin Adımları

Script aşağıdaki adımları otomatik olarak gerçekleştirir:

1. **Dosya Kontrolü**: Tüm model dosyalarının varlığını kontrol eder
2. **Repository Oluşturma**: Hugging Face Hub'da repository oluşturur
3. **Dosya Hazırlama**: Upload için geçici dizin hazırlar
4. **Model Card**: README.md dosyasını hazırlar
5. **Upload**: Tüm dosyaları yükler
6. **Temizlik**: Geçici dosyaları temizler

## 🎯 Beklenen Çıktı

Başarılı upload sonrası şu çıktıyı göreceksiniz:

```
🎉 UPLOAD COMPLETED SUCCESSFULLY!
==================================================
📁 Repository: your-username/ai-learning-psychology-analyzer
🔗 URL: https://huggingface.co/your-username/ai-learning-psychology-analyzer
📊 Files: 13
🕒 Updated: 2024-01-XX

🎯 Next Steps:
  1. Visit your repository on Hugging Face
  2. Check that all files uploaded correctly
  3. Update any personal information in the model card
  4. Test the model downloads with the provided code examples
```

## 🔧 Model Kullanımı

### Temel Kullanım
```python
from huggingface_inference import LearningPsychologyAnalyzer

# Analyzer'ı başlatın
analyzer = LearningPsychologyAnalyzer("your-username/ai-learning-psychology-analyzer")

# Öğrenci verilerini analiz edin
student_data = {
    'attention_features': [0.8, 0.7, 0.9, 0.6, 0.8],
    'cognitive_features': [0.6, 0.4, 0.7, 0.5, 0.3],
    'style_features': [0.7, 0.6, 0.8, 0.9, 0.5]
}

results = analyzer.analyze_student(student_data)
print(results)
```

### Sadece Hugging Face Hub
```python
from huggingface_hub import hf_hub_download
import pickle

# Model indir
model_path = hf_hub_download(
    repo_id="your-username/ai-learning-psychology-analyzer",
    filename="attention_tracker_model.pkl"
)

# Model yükle
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Tahmin yap
prediction = model.predict([[0.8, 0.7, 0.9, 0.6, 0.8]])
```

## 🐛 Sorun Giderme

### Yaygın Hatalar

1. **Authentication Error**
   ```
   ❌ Error: Token is required
   ```
   **Çözüm**: API token'ınızı kontrol edin

2. **File Not Found**
   ```
   ❌ Model file not found
   ```
   **Çözüm**: Modelleri yeniden eğitin: `python train_models_real_data.py`

3. **Permission Denied**
   ```
   ❌ Repository creation failed
   ```
   **Çözüm**: Token'ınızın "Write" yetkisi olduğunu kontrol edin

### Debug Modu
```bash
python upload_to_huggingface.py --username your-username --token your-token --dry-run
```

## 📱 Model Paylaşımı

### Public Repository
Modeliniz varsayılan olarak public (herkese açık) olacak.

### Private Repository
Private repository için Hugging Face Pro hesabı gereklidir.

### Model Card Düzenleme
Upload sonrası model kartınızı düzenlemek için:
1. Repository sayfasına gidin
2. "Edit model card" butonuna tıklayın
3. README.md dosyasını düzenleyin

## 🎯 Sonraki Adımlar

1. **Repository'nizi ziyaret edin**: https://huggingface.co/your-username/ai-learning-psychology-analyzer
2. **Model kartını güncelleyin**: Kişisel bilgilerinizi ekleyin
3. **Demo oluşturun**: Hugging Face Spaces ile interaktif demo
4. **Inference API'yi test edin**: Model tahminlerini test edin

## 📚 Faydalı Linkler

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
- [Inference API](https://huggingface.co/docs/api-inference/index)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)

## 🤝 Topluluk

- [Hugging Face Discord](https://discord.gg/hugging-face)
- [Hugging Face Forum](https://discuss.huggingface.co/)
- [GitHub Issues](https://github.com/huggingface/huggingface_hub/issues)

---

**İyi şanslar! 🚀 Modelinizi dünyayla paylaşın ve eğitim teknolojilerine katkıda bulunun!** 