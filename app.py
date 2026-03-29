import streamlit as st
import numpy as np
import pickle
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import arabic_reshaper
from bidi.algorithm import get_display

nltk.download('stopwords', quiet=True)
matplotlib.rcParams['font.family'] = 'Arial'

def ar(text):
    return get_display(arabic_reshaper.reshape(str(text)))

# ════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1),  initializer="zeros",          trainable=True)
        super().build(input_shape)
    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

@st.cache_resource
def load_assets():
    m = load_model('model/sentiment_model.keras', custom_objects={'SelfAttention': SelfAttention})
    with open('model/tokenizer.pkl', 'rb') as f: tok = pickle.load(f)
    with open('model/label_encoder.pkl', 'rb') as f: le = pickle.load(f)
    return m, tok, le

model, tokenizer, le = load_assets()
MAX_LEN = 150

# ════════════════════════════════════════════════════
# Text Processing & Smart Analysis
# ════════════════════════════════════════════════════

arabic_stopwords = set(stopwords.words('arabic'))
arabic_stopwords -= {'لا','ما','لم','ليس','لن','غير','بدون','لو','مو'}
arabic_stopwords.update({'تم','ان','كان','يكون','عن','بعد','قبل'})

POSITIVE_KEYWORDS = {
    'ممتاز','رائع','جميل','مميز','احسن','أحسن','مبدع','شكرا','شكراً','ممنون',
    'يسلم','بارك','تسلم','عظيم','مدهش','مفيد','نافع','ينصح','أنصح',
    'حلو','حلوة','سريع','دقيق','نظيف','أفضل','افضل','ناجح','محترم','كفء',
    'سهل','بسيط','لذيذ','طيب','زين','واو','ابداع','جيد','كويس','مريح',
    'راحة','راضي','سعيد','مبسوط','ثقة','موثوق','ممتازة','جيدة','رائعة'
}
NEGATIVE_KEYWORDS = {
    'سيء','سيئ','مزعج','ضعيف','ردي','رديء','مخيب','احتيال','كذب','غش',
    'بطيء','مأساة','فاشل','تعبان','رهيب','مقرف','وحش','خايس','احباط',
    'خيبة','مشكلة','عطل','اعطال','مكسور','بائس','متعب','مهمل','متأخر',
    'غلط','خطأ','خسارة','مضيعة','مزيف','نصب','كارثة','سلبي','وحشة',
    'مره','زفت','سيئة','ضعيفة','رديئة','بطيئة'
}

def is_arabic(text):
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', str(text)))
    return arabic_chars > len(str(text)) * 0.2

def clean(text):
    text = re.sub(r"http\S+|www\S+|@\S+|#", "", str(text))
    text = re.sub(r"[\u064B-\u0652]", "", text)
    for a, b in [("[إأآا]","ا"),("ى","ي"),("ة","ه"),("ؤ","ء"),("ئ","ء")]:
        text = re.sub(a, b, text)
    text = re.sub(r"[^\u0621-\u064A\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def remove_sw(text):
    return ' '.join(w for w in text.split() if w not in arabic_stopwords)

def get_smart_label(class_idx, score):
    if class_idx == 2:
        if score >= 0.85:   return "إيجابي قوي جداً 🔥", "🟢"
        elif score >= 0.65: return "إيجابي 😊", "🟢"
        else:               return "إيجابي ضعيف 🙂", "🟡"
    elif class_idx == 0:
        if score >= 0.85:   return "سلبي قوي 😡", "🔴"
        elif score >= 0.65: return "سلبي 😞", "🔴"
        else:               return "سلبي ضعيف 😕", "🟠"
    else:
        if score >= 0.80:   return "محايد تماماً 😐", "🟡"
        else:               return "محايد 😐", "🟡"

def get_confidence_label(score):
    if score >= 0.90:   return "عالية جداً 🎯", "#2d6a4f"
    elif score >= 0.75: return "عالية ✅",       "#52b788"
    elif score >= 0.60: return "متوسطة ⚠️",      "#c9972a"
    else:               return "منخفضة ❓",       "#e76f51"

def extract_keywords(original_text, class_idx):
    words     = original_text.split()
    norm_words= [clean(w) for w in words]
    kw_set = POSITIVE_KEYWORDS if class_idx == 2 else \
             NEGATIVE_KEYWORDS if class_idx == 0 else \
             POSITIVE_KEYWORDS | NEGATIVE_KEYWORDS
    found = []
    for orig, norm in zip(words, norm_words):
        if norm in kw_set and orig not in found:
            found.append(orig)
    for w in remove_sw(clean(original_text)).split():
        if w in kw_set and w not in found:
            found.append(w)
    return found[:6]

def get_reason(original_text, class_idx, keywords):
    if keywords:
        kw_str = "، ".join(f'"{k}"' for k in keywords[:3])
        if class_idx == 2: return f"يحتوي النص على كلمات إيجابية مثل {kw_str}"
        elif class_idx == 0: return f"يحتوي النص على كلمات سلبية مثل {kw_str}"
        else: return f"النص يحتوي كلمات متوازنة مثل {kw_str}"
    else:
        if class_idx == 2: return "الأسلوب العام للنص يعطي انطباعاً إيجابياً"
        elif class_idx == 0: return "الأسلوب العام للنص يعطي انطباعاً سلبياً"
        else: return "النص لا يميل بوضوح نحو الإيجابية أو السلبية"

def predict(text):
    c   = remove_sw(clean(text))
    seq = tokenizer.texts_to_sequences([c])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    p   = model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(p))
    return idx, float(p[idx]), p

# ════════════════════════════════════════════════════
# Shared Renderer (CSV tab)
# ════════════════════════════════════════════════════

COLORS_CHART = ['#2d6a4f','#e76f51','#e9c46a']
DARK_BG      = '#1e2d26'

def render_results(comments, source_label="التعليق"):
    with st.spinner("⏳ جاري تحليل التعليقات..."):
        raw = [predict(str(c)) for c in comments]

    rows = []
    for c, (idx, score, probs) in zip(comments, raw):
        smart_lbl, _ = get_smart_label(idx, score)
        rows.append({source_label: c, 'التصنيف': smart_lbl,
                     'نسبة الثقة %': round(score*100,1), '_class': idx})
    df_r = pd.DataFrame(rows)

    def base_class(lbl):
        if 'إيجابي' in lbl: return 'إيجابي 😊'
        if 'سلبي'   in lbl: return 'سلبي 😞'
        return 'محايد 😐'

    df_r['_base'] = df_r['التصنيف'].apply(base_class)
    counts = df_r['_base'].value_counts()
    total  = len(df_r)

    st.success(f"✅ اكتمل تحليل {total} تعليق بنجاح!")

    # ── Metrics ──
    st.markdown('<p class="section-title">📊 الإحصائيات</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    def mc(col, emoji, label, count):
        pct = f"{count/total*100:.1f}%"
        col.markdown(f"""
            <div style='background:white;border-radius:16px;padding:22px 18px;
                        border:1.5px solid #c8e6d4;text-align:center;
                        box-shadow:0 3px 16px rgba(45,106,79,0.08);'>
                <div style='color:#111;font-size:1rem;font-weight:700;margin-bottom:8px;'>{emoji} {label}</div>
                <div style='color:#1a3a2e;font-size:2.4rem;font-weight:800;line-height:1;'>{count}</div>
                <div style='color:#2d6a4f;font-size:0.95rem;font-weight:700;margin-top:6px;'>{pct}</div>
            </div>
        """, unsafe_allow_html=True)
    mc(c1,"🟢","إيجابي",counts.get('إيجابي 😊',0))
    mc(c2,"🔴","سلبي",  counts.get('سلبي 😞',  0))
    mc(c3,"🟡","محايد", counts.get('محايد 😐', 0))

    # ── Best & Worst ──
    pos_df = df_r[df_r['_class']==2]
    neg_df = df_r[df_r['_class']==0]
    if not pos_df.empty or not neg_df.empty:
        st.markdown('<p class="section-title">🏆 أبرز التعليقات</p>', unsafe_allow_html=True)
        bw1, bw2 = st.columns(2, gap="medium")
        if not pos_df.empty:
            best = pos_df.loc[pos_df['نسبة الثقة %'].idxmax(), source_label]
            bw1.markdown(f"""
                <div style='background:#eaf5ee;border-radius:14px;padding:18px 20px;
                            border:1.5px solid #b7dfc8;min-height:100px;'>
                    <div style='color:#2d6a4f;font-weight:800;font-size:0.95rem;margin-bottom:8px;'>
                        🥇 أكثر تعليق إيجابي
                    </div>
                    <div style='color:#1a3a2e;font-size:0.93rem;line-height:1.7;
                                font-style:italic;direction:rtl;'>
                        "{str(best)[:200]}{'...' if len(str(best))>200 else ''}"
                    </div>
                </div>
            """, unsafe_allow_html=True)
        if not neg_df.empty:
            worst = neg_df.loc[neg_df['نسبة الثقة %'].idxmax(), source_label]
            bw2.markdown(f"""
                <div style='background:#fdf0ec;border-radius:14px;padding:18px 20px;
                            border:1.5px solid #f5c4b5;min-height:100px;'>
                    <div style='color:#e76f51;font-weight:800;font-size:0.95rem;margin-bottom:8px;'>
                        ⚠️ أكثر تعليق سلبي
                    </div>
                    <div style='color:#1a3a2e;font-size:0.93rem;line-height:1.7;
                                font-style:italic;direction:rtl;'>
                        "{str(worst)[:200]}{'...' if len(str(worst))>200 else ''}"
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # ── Filter + Table ──
    st.markdown('<p class="section-title">📋 النتائج التفصيلية</p>', unsafe_allow_html=True)
    filter_opts = ["الكل 🔎","إيجابي فقط 🟢","سلبي فقط 🔴","محايد فقط 🟡"]
    chosen = st.selectbox("🔽 فلترة النتائج:", filter_opts, key=f"filter_{id(comments)}")
    ddf = df_r[[source_label,'التصنيف','نسبة الثقة %']].copy()
    if   chosen == "إيجابي فقط 🟢": ddf = ddf[df_r['_class']==2]
    elif chosen == "سلبي فقط 🔴":   ddf = ddf[df_r['_class']==0]
    elif chosen == "محايد فقط 🟡":  ddf = ddf[df_r['_class']==1]
    st.caption(f"📌 عدد النتائج المعروضة: {len(ddf)} من {total}")
    st.dataframe(ddf, use_container_width=True, height=320)

    # ── Charts ──
    st.markdown('<p class="section-title">📈 الرسوم البيانية</p>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13,5), facecolor=DARK_BG)
    labels_ar = [ar(l) for l in counts.index]

    axes[0].set_facecolor(DARK_BG)
    bars = axes[0].bar(labels_ar, counts.values, color=COLORS_CHART[:len(counts)],
                       width=0.45, edgecolor='none', zorder=3)
    axes[0].set_title(ar('توزيع التصنيفات'), fontsize=14, color='#f5f2e8', pad=14)
    axes[0].set_xlabel(ar('التصنيف'), color='#95d5b2', labelpad=8)
    axes[0].set_ylabel(ar('العدد'),   color='#95d5b2', labelpad=8)
    axes[0].tick_params(colors='#c8e6d4')
    axes[0].grid(axis='y', color='#2d6a4f', alpha=0.3, zorder=0)
    for sp in axes[0].spines.values(): sp.set_color('#2d4a38')
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.3, str(v),
                     ha='center', color='white', fontweight='bold', fontsize=12)

    axes[1].set_facecolor(DARK_BG)
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels_ar, colors=COLORS_CHART[:len(counts)],
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': DARK_BG, 'linewidth': 3})
    for t in texts:     t.set_color('#c8e6d4'); t.set_fontsize(11); t.set_fontweight('600')
    for t in autotexts: t.set_color('white');  t.set_fontweight('bold'); t.set_fontsize(11)
    axes[1].set_title(ar('النسب المئوية'), fontsize=14, color='#f5f2e8', pad=14)
    plt.tight_layout(pad=2.5)
    st.pyplot(fig)

    # ── Download ──
    csv_out = ddf.to_csv(index=False, encoding='utf-8-sig')
    st.download_button("⬇️ تحميل النتائج CSV", data=csv_out,
                       file_name="sentiment_results.csv", mime="text/csv")


# ════════════════════════════════════════════════════
# Page Config & CSS
# ════════════════════════════════════════════════════

st.set_page_config(page_title="مِيزان | تحليل المشاعر", page_icon="🌿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&display=swap');
*,*::before,*::after{font-family:'Tajawal',sans-serif !important;box-sizing:border-box;}

.stApp{background:linear-gradient(145deg,#f5f2e8 0%,#eaf5ee 50%,#f2ede0 100%) !important;min-height:100vh;}
section[data-testid="stSidebar"]{display:none;}

.main-header{background:linear-gradient(135deg,#1a3a2e 0%,#2d6a4f 65%,#52b788 100%);
    border-radius:22px;padding:44px 40px 36px;text-align:center;margin-bottom:24px;
    box-shadow:0 10px 40px rgba(45,106,79,0.22);position:relative;overflow:hidden;}
.main-header::after{content:'';position:absolute;top:-40px;right:-40px;
    width:200px;height:200px;background:rgba(255,255,255,0.05);border-radius:50%;}
.badge{display:inline-block;background:rgba(255,255,255,0.15);color:#c7ebd5;
    border:1px solid rgba(255,255,255,0.2);border-radius:20px;
    padding:5px 18px;font-size:0.82rem;font-weight:500;margin-bottom:14px;}
.main-header h1{color:#f5f2e8 !important;font-size:2.8rem !important;font-weight:800 !important;margin:0 0 10px !important;}
.main-header p{color:#95d5b2 !important;font-size:1.05rem !important;margin:0 !important;}

.how-to{background:white;border-radius:14px;padding:16px 24px;border:1.5px solid #c8e6d4;
    margin-bottom:24px;box-shadow:0 2px 12px rgba(45,106,79,0.06);
    display:flex;gap:28px;justify-content:center;flex-wrap:wrap;}
.how-to-step{display:flex;align-items:center;gap:10px;color:#1a3a2e;font-weight:600;font-size:0.93rem;}
.step-num{background:#2d6a4f;color:white;border-radius:50%;width:26px;height:26px;
    display:flex;align-items:center;justify-content:center;font-weight:800;font-size:0.85rem;flex-shrink:0;}

.stTabs [data-baseweb="tab-list"]{background:#dff0e6 !important;border-radius:14px !important;
    padding:5px !important;gap:4px !important;border:1.5px solid #b7dfc8 !important;flex-wrap:wrap !important;}
.stTabs [data-baseweb="tab"]{border-radius:10px !important;color:#2d6a4f !important;
    font-weight:600 !important;font-size:0.93rem !important;padding:10px 18px !important;
    transition:all 0.2s !important;background:transparent !important;}
.stTabs [aria-selected="true"]{background:#2d6a4f !important;color:white !important;
    box-shadow:0 3px 14px rgba(45,106,79,0.28) !important;}

.section-title{color:#1a3a2e;font-size:1.1rem;font-weight:700;margin:28px 0 14px;
    padding-right:14px;border-right:4px solid #52b788;}

.kw-tag{display:inline-block;background:#eaf5ee;color:#1a3a2e;border:1.5px solid #b7dfc8;
    border-radius:20px;padding:4px 14px;font-size:0.9rem;font-weight:700;margin:3px;}

[data-testid="stFileUploader"]{background:white !important;border-radius:14px !important;
    border:2px dashed #52b788 !important;padding:10px !important;}
[data-testid="stFileUploader"] p,[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small{color:#2d6a4f !important;font-weight:500 !important;}

[data-baseweb="select"]>div{background:white !important;border-radius:11px !important;
    border:1.5px solid #52b788 !important;color:#1a2e24 !important;
    font-size:1rem !important;min-height:46px !important;}
[data-baseweb="select"] span,[data-baseweb="select"] div{color:#1a2e24 !important;font-weight:500 !important;}
label[data-testid="stWidgetLabel"]>div>p{color:#1a3a2e !important;font-weight:700 !important;}

[data-testid="stSlider"]{background:white !important;border-radius:14px !important;
    padding:18px 20px 14px !important;border:1.5px solid #c8e6d4 !important;}
[data-testid="stSlider"] p{color:#1a3a2e !important;font-weight:700 !important;}
[data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{color:#2d6a4f !important;font-weight:600 !important;}

.stButton>button{background:linear-gradient(135deg,#2d6a4f,#1a3a2e) !important;color:white !important;
    border:none !important;border-radius:12px !important;padding:13px 32px !important;
    font-size:1.02rem !important;font-weight:700 !important;width:100% !important;
    transition:all 0.25s !important;box-shadow:0 4px 18px rgba(45,106,79,0.28) !important;}
.stButton>button:hover{transform:translateY(-2px) !important;
    box-shadow:0 8px 28px rgba(45,106,79,0.38) !important;
    background:linear-gradient(135deg,#52b788,#2d6a4f) !important;}

[data-testid="stDataFrame"]{border-radius:14px !important;overflow:hidden !important;
    border:1.5px solid #c8e6d4 !important;background:white !important;}
[data-testid="stDownloadButton"]>button{background:white !important;color:#2d6a4f !important;
    border:2px solid #52b788 !important;border-radius:11px !important;
    font-weight:700 !important;padding:10px 24px !important;}
[data-testid="stDownloadButton"]>button:hover{background:#eaf5ee !important;}

[data-testid="stAlert"]{border-radius:12px !important;font-weight:500 !important;}

textarea{background:white !important;border-radius:12px !important;
    border:1.5px solid #52b788 !important;color:#1a2e24 !important;
    font-size:1rem !important;line-height:1.7 !important;}
textarea:focus{border-color:#2d6a4f !important;box-shadow:0 0 0 3px rgba(82,183,136,0.18) !important;}

input[type="text"],input[type="password"]{background:white !important;border-radius:11px !important;
    border:1.5px solid #52b788 !important;color:#1a2e24 !important;
    font-size:1rem !important;padding:10px 14px !important;}

.stSpinner>div{border-top-color:#2d6a4f !important;}
[data-testid="stCaptionContainer"] p{color:#2d6a4f !important;font-weight:600 !important;}

@media(max-width:768px){
    .main-header h1{font-size:1.8rem !important;}
    .main-header{padding:28px 20px 24px;}
    .how-to{gap:16px;}
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# Header
# ════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <div class="badge">🌿 نموذج ذكاء اصطناعي متخصص بالعربية</div>
    <h1>🧠 مِيزان لتحليل المشاعر</h1>
    <p>حلّل مراجعاتك العربية بدقة — إيجابي، سلبي، أو محايد</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="how-to">
    <div class="how-to-step"><div class="step-num">1</div><span>اختر طريقة الإدخال من التبويبات</span></div>
    <div class="how-to-step"><div class="step-num">2</div><span>أدخل النص أو ارفع ملف CSV</span></div>
    <div class="how-to-step"><div class="step-num">3</div><span>اضغط تحليل وشاهد النتائج فوراً</span></div>
    <div class="how-to-step"><div class="step-num">4</div><span>حمّل النتائج بصيغة CSV</span></div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📄  CSV",
    "✍️  نص واحد",
    "📝  تحليل متعدد",
])

# ════════════════════════════════════════════════════
# Tab 1 — CSV
# ════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader("📂 ارفع ملف CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown('<p class="section-title">👀 معاينة الملف</p>', unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        st.markdown('<p class="section-title">⚙️ إعدادات التحليل</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns([1,1], gap="large")
        with col_a:
            col_sel = st.selectbox("🔽 اختر عمود المراجعات:", df.columns.tolist())
        with col_b:
            n_rows = st.slider("🔢 عدد المراجعات:", min_value=5,
                               max_value=len(df), value=min(50,len(df)), step=5)
        st.markdown(f"""
            <div style='background:#eaf5ee;border-radius:12px;padding:14px 20px;
                        border-right:4px solid #52b788;color:#1a3a2e;font-weight:600;
                        font-size:0.97rem;margin:16px 0;border:1px solid #b7dfc8;'>
                📌 سيتم تحليل أول <span style='color:#2d6a4f;font-size:1.1rem;'><b>{n_rows}</b></span>
                مراجعة من أصل <span style='color:#2d6a4f;'><b>{len(df)}</b></span>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 ابدأ التحليل الآن", key="csv_btn"):
            df_sub = df.head(n_rows).dropna(subset=[col_sel]).copy()
            render_results(df_sub[col_sel].tolist(), source_label=col_sel)

# ════════════════════════════════════════════════════
# Tab 2 — Single Text
# ════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">✍️ أدخل النص للتحليل</p>', unsafe_allow_html=True)
    text = st.text_area("", height=150,
                        placeholder="مثال: خدمة الشركة ممتازة والتوصيل كان سريع جداً وأنصح بها...",
                        label_visibility="collapsed", key="single_text")

    if st.button("🔍 تحليل النص", key="single_btn"):
        if not text.strip():
            st.warning("⚠️ من فضلك أدخل نصاً أولاً.")
        elif not is_arabic(text):
            st.error("⚠️ هذا النص ليس عربياً. مِيزان متخصص في اللغة العربية فقط.")
        else:
            with st.spinner("🔍 جاري تحليل النص..."):
                idx, score, probs = predict(text)
                smart_label, dot  = get_smart_label(idx, score)
                conf_label, conf_color = get_confidence_label(score)
                keywords = extract_keywords(text, idx)
                reason   = get_reason(text, idx, keywords)

            if idx == 2:   color, bg, border = '#2d6a4f','#eaf5ee','#b7dfc8'
            elif idx == 0: color, bg, border = '#e76f51','#fdf0ec','#f5c4b5'
            else:          color, bg, border = '#c9972a','#fdf8ec','#f0dfa0'

            st.markdown(f"""
                <div style='background:{bg};border-radius:20px;padding:36px 28px;
                            text-align:center;border:2px solid {border};
                            box-shadow:0 6px 30px rgba(0,0,0,0.08);margin:16px 0 20px;'>
                    <div style='font-size:3rem;margin-bottom:10px;line-height:1;'>{dot}</div>
                    <div style='color:{color};font-size:2.1rem;font-weight:800;margin-bottom:16px;'>
                        {smart_label}
                    </div>
                    <div style='display:flex;justify-content:center;gap:36px;flex-wrap:wrap;'>
                        <div>
                            <div style='color:#555;font-size:0.85rem;font-weight:600;margin-bottom:4px;'>نسبة الثقة</div>
                            <div style='color:{color};font-size:1.7rem;font-weight:800;'>{score*100:.1f}%</div>
                        </div>
                        <div>
                            <div style='color:#555;font-size:0.85rem;font-weight:600;margin-bottom:4px;'>مستوى الثقة</div>
                            <div style='color:{conf_color};font-size:1.1rem;font-weight:700;'>{conf_label}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style='background:white;border-radius:14px;padding:20px 22px;
                            border:1.5px solid #c8e6d4;margin-bottom:12px;
                            box-shadow:0 2px 12px rgba(45,106,79,0.06);'>
                    <div style='color:#1a3a2e;font-weight:800;font-size:0.97rem;margin-bottom:8px;'>
                        📌 سبب التقييم
                    </div>
                    <div style='color:#333;font-size:0.95rem;line-height:1.7;direction:rtl;'>{reason}</div>
                </div>
            """, unsafe_allow_html=True)

            if keywords:
                kw_html = "".join(f'<span class="kw-tag">{kw}</span>' for kw in keywords)
                st.markdown(f"""
                    <div style='background:white;border-radius:14px;padding:18px 22px;
                                border:1.5px solid #c8e6d4;margin-bottom:20px;
                                box-shadow:0 2px 12px rgba(45,106,79,0.06);'>
                        <div style='color:#1a3a2e;font-weight:800;font-size:0.97rem;margin-bottom:10px;'>
                            ✨ الكلمات المؤثرة
                        </div>
                        <div style='direction:rtl;'>{kw_html}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown('<p class="section-title">📊 توزيع الاحتمالات</p>', unsafe_allow_html=True)
            for lab, val, col in [
                ('إيجابي 😊', float(probs[2]), '#2d6a4f'),
                ('محايد 😐',  float(probs[1]), '#c9972a'),
                ('سلبي 😞',   float(probs[0]), '#e76f51'),
            ]:
                st.markdown(f"""
                    <div style='background:white;border-radius:12px;padding:14px 20px;
                                border:1.5px solid #c8e6d4;margin-bottom:10px;
                                box-shadow:0 2px 8px rgba(45,106,79,0.05);'>
                        <div style='display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:8px;'>
                            <span style='color:#111;font-weight:700;font-size:1rem;'>{lab}</span>
                            <span style='color:{col};font-weight:800;font-size:1.1rem;'>{val*100:.1f}%</span>
                        </div>
                        <div style='background:#e8f5ee;border-radius:8px;height:12px;overflow:hidden;'>
                            <div style='width:{val*100:.1f}%;background:{col};border-radius:8px;height:12px;'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 تحليل نص جديد", key="reset_btn"):
                st.rerun()

# ════════════════════════════════════════════════════
# Tab 3 — Multi-Text
# ════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">📝 تحليل نصوص متعددة دفعة واحدة</p>', unsafe_allow_html=True)
    st.markdown("""
        <div style='background:#eaf5ee;border-radius:12px;padding:14px 20px;
                    border:1px solid #b7dfc8;border-right:4px solid #52b788;
                    color:#1a3a2e;font-weight:500;font-size:0.93rem;margin-bottom:16px;'>
            💡 الصق عدة جمل أو تعليقات — <b>كل سطر = تعليق واحد</b>
        </div>
    """, unsafe_allow_html=True)

    multi_text = st.text_area("", height=220,
                              placeholder="الخدمة كانت رائعة جداً\nبطيء جداً ومزعج\nعادي مو بس\nأنصح بشدة بهذا المنتج\nتجربة سيئة لن أكررها",
                              label_visibility="collapsed", key="multi_text")

    if st.button("🚀 تحليل جميع النصوص", key="multi_btn"):
        lines = [l.strip() for l in multi_text.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("⚠️ من فضلك أدخل نصوصاً أولاً (كل سطر = تعليق).")
        else:
            non_ar   = [l for l in lines if not is_arabic(l)]
            ar_lines = [l for l in lines if is_arabic(l)]
            if non_ar: st.warning(f"⚠️ تم تجاهل {len(non_ar)} سطر غير عربي.")
            if not ar_lines:
                st.error("❌ لا توجد نصوص عربية للتحليل.")
            else:
                with st.spinner(f"⏳ جاري تحليل {len(ar_lines)} نص..."):
                    rows = []
                    for line in ar_lines:
                        idx, score, probs = predict(line)
                        smart_lbl, _     = get_smart_label(idx, score)
                        kws              = extract_keywords(line, idx)
                        rows.append({'النص': line, 'التصنيف': smart_lbl,
                                     'نسبة الثقة %': round(score*100,1),
                                     'الكلمات المؤثرة': ' ، '.join(kws) if kws else '—',
                                     '_class': idx})
                df_m = pd.DataFrame(rows)
                base_map = {2:'إيجابي 😊', 0:'سلبي 😞', 1:'محايد 😐'}
                counts   = df_m['_class'].map(base_map).value_counts()
                total    = len(df_m)

                st.success(f"✅ تم تحليل {total} نص بنجاح!")

                st.markdown('<p class="section-title">📊 الإحصائيات</p>', unsafe_allow_html=True)
                mc1, mc2, mc3 = st.columns(3, gap="medium")
                def mcard(col, emoji, label, count):
                    pct = f"{count/total*100:.1f}%"
                    col.markdown(f"""
                        <div style='background:white;border-radius:16px;padding:22px 18px;
                                    border:1.5px solid #c8e6d4;text-align:center;
                                    box-shadow:0 3px 16px rgba(45,106,79,0.08);'>
                            <div style='color:#111;font-size:1rem;font-weight:700;margin-bottom:8px;'>{emoji} {label}</div>
                            <div style='color:#1a3a2e;font-size:2.4rem;font-weight:800;line-height:1;'>{count}</div>
                            <div style='color:#2d6a4f;font-size:0.95rem;font-weight:700;margin-top:6px;'>{pct}</div>
                        </div>
                    """, unsafe_allow_html=True)
                mcard(mc1,"🟢","إيجابي",counts.get('إيجابي 😊',0))
                mcard(mc2,"🔴","سلبي",  counts.get('سلبي 😞',  0))
                mcard(mc3,"🟡","محايد", counts.get('محايد 😐', 0))

                st.markdown('<p class="section-title">📋 النتائج لكل نص</p>', unsafe_allow_html=True)
                for _, row in df_m.iterrows():
                    if row['_class']==2:   c,bg,bd='#2d6a4f','#eaf5ee','#b7dfc8'
                    elif row['_class']==0: c,bg,bd='#e76f51','#fdf0ec','#f5c4b5'
                    else:                  c,bg,bd='#c9972a','#fdf8ec','#f0dfa0'
                    kw_html = "".join(f'<span class="kw-tag">{k}</span>'
                                      for k in row['الكلمات المؤثرة'].split(' ، ')
                                      if k and k!='—')
                    st.markdown(f"""
                        <div style='background:{bg};border-radius:14px;padding:16px 20px;
                                    border:1.5px solid {bd};margin-bottom:10px;
                                    box-shadow:0 2px 10px rgba(0,0,0,0.05);'>
                            <div style='display:flex;justify-content:space-between;
                                        align-items:flex-start;gap:10px;flex-wrap:wrap;'>
                                <div style='color:#1a3a2e;font-size:0.97rem;font-weight:500;
                                            flex:1;direction:rtl;line-height:1.6;'>
                                    {row['النص']}
                                </div>
                                <div style='text-align:center;min-width:110px;'>
                                    <div style='color:{c};font-weight:800;font-size:1rem;'>{row['التصنيف']}</div>
                                    <div style='color:#555;font-size:0.85rem;font-weight:600;'>{row['نسبة الثقة %']}%</div>
                                </div>
                            </div>
                            {f"<div style='margin-top:10px;direction:rtl;'>{kw_html}</div>" if kw_html else ""}
                        </div>
                    """, unsafe_allow_html=True)

                csv_out = df_m[['النص','التصنيف','نسبة الثقة %','الكلمات المؤثرة']].to_csv(
                    index=False, encoding='utf-8-sig')
                st.download_button("⬇️ تحميل النتائج CSV", data=csv_out,
                                   file_name="multi_sentiment.csv", mime="text/csv")