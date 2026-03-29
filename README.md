# 🧠 مِيزان | Arabic Sentiment Analysis
> تحليل مشاعر المراجعات العربية بالذكاء الاصطناعي

[English](#english) | [العربية](#arabic)

---

## English

### Overview
**Mizan** is an AI-powered Arabic sentiment analysis application that classifies text into three categories: **Positive 😊 | Negative 😞 | Neutral 😐**

Built with a **Bidirectional LSTM + Self-Attention** deep learning model and deployed via **Streamlit**.

### ✨ Features
- 📄 **Bulk Analysis** — Upload a CSV file and analyze hundreds of reviews at once
- ✍️ **Single Text** — Analyze any Arabic sentence instantly
- 📊 **Visual Reports** — Bar charts, pie charts, and confidence scores
- ⬇️ **Export Results** — Download analyzed results as CSV
- 🌿 **Clean UI** — Elegant green & cream themed interface

### 🧠 Model Architecture
| Component | Details |
|---|---|
| Embedding | 60,000 vocab — 64 dims |
| Encoder | Bidirectional LSTM (64 units) |
| Attention | Custom Self-Attention Layer |
| Regularization | BatchNorm + Dropout (0.4) |
| Output | Softmax — 3 classes |

### 📊 Model Performance
| Metric | Score |
|---|---|
| Test Accuracy | **83.62%** |
| Macro F1-Score | **0.83** |
| Positive F1 | 0.84 |
| Negative F1 | 0.76 |
| Neutral F1 | 0.87 |

### 📁 Project Structure
```text
arabic-sentiment-analysis/
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
├── README.md
├── .gitignore
├── model/
│   ├── sentiment_model.keras
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
└── notebooks/              # Training notebooks
```

### 🚀 Run Locally
```bash
git clone https://github.com/your-username/arabic-sentiment-analysis.git
cd arabic-sentiment-analysis
pip install -r requirements.txt
python -m streamlit run app.py
```

### 🛠️ Tech Stack
- Python 3.10+
- TensorFlow / Keras
- Streamlit
- NLTK — Arabic stopwords
- arabic-reshaper + python-bidi
- Scikit-learn
- Matplotlib / Seaborn

---

### 👤 Developer
Made with ❤️ by **Abdulrahman Alruhaili**
---

## Arabic

### نظرة عامة
**مِيزان** تطبيق ذكاء اصطناعي لتحليل مشاعر النصوص العربية، يصنّف المراجعات إلى ثلاث فئات:
**إيجابي 😊 | سلبي 😞 | محايد 😐**

مبني على نموذج **Bidirectional LSTM + Self-Attention** ومنشور عبر **Streamlit**.

### ✨ المميزات
- 📄 **تحليل ملفات CSV** — حلّل مئات المراجعات دفعة واحدة
- ✍️ **تحليل نص واحد** — أدخل أي جملة واحصل على النتيجة فوراً
- 📊 **تقارير بصرية** — رسوم بيانية ونسب ثقة لكل تصنيف
- ⬇️ **تصدير النتائج** — حمّل النتائج بصيغة CSV
- 🌿 **واجهة أنيقة** — تصميم أخضر وسكري متناسق

### 📊 نتائج النموذج
| المقياس | النتيجة |
|---|---|
| دقة الاختبار | **83.62%** |
| Macro F1 | **0.83** |
| F1 إيجابي | 0.84 |
| F1 سلبي | 0.76 |
| F1 محايد | 0.87 |

### 🚀 تشغيل محلي
```bash
git clone https://github.com/your-username/arabic-sentiment-analysis.git
cd arabic-sentiment-analysis
pip install -r requirements.txt
python -m streamlit run app.py
```

### 👤 المطوّر
صُنع بـ ❤️ بواسطة **عبدالرحمن الرحيلي**

---

> ⭐ إذا أعجبك المشروع، لا تنسى تضغط Star على GitHub!