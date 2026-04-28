import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="منصة التحليل الذكي", layout="wide")
st.title("📊 منصة التحليل الإحصائي الذكي (PCA)")

uploaded_file = st.file_uploader("ارفع ملف البيانات (CSV)", type="csv")

if uploaded_file:
    # قراءة البيانات مع التعامل مع الفاصلة المنقوطة
    df = pd.read_csv(uploaded_file, sep=';')
    
    # تنظيف البيانات: تحويل كل شيء لأرقام وحذف الأسطر الفارغة
    numeric_df = df.select_dtypes(include=['number']).dropna()
    
    if not numeric_df.empty and numeric_df.shape[0] > 2:
        st.success(f"تمت معالجة بيانات {numeric_df.shape[0]} طالب بنجاح!")
        
        # تحضير البيانات للـ PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)
        
        # تقسيم الشاشة لعرض الرسمين بجانب بعض
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1️⃣ خريطة الطلاب (Individuals)")
            fig1, ax1 = plt.subplots()
            ax1.scatter(components[:, 0], components[:, 1], c='skyblue', edgecolors='navy')
            for i in range(len(components)):
                ax1.text(components[i, 0], components[i, 1], str(i+1), fontsize=9)
            ax1.set_xlabel("المحور 1")
            ax1.set_ylabel("المحور 2")
            ax1.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig1)

        with col2:
            st.subheader("2️⃣ دائرة الارتباطات (Variables)")
            fig2, ax2 = plt.subplots()
            for i, var in enumerate(numeric_df.columns):
                ax2.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
                         head_width=0.05, color='red', alpha=0.8)
                ax2.text(pca.components_[0, i]*1.15, pca.components_[1, i]*1.15, var, color='darkred', fontweight='bold')
            
            circle = plt.Circle((0,0), 1, color='blue', fill=False, linestyle='--')
            ax2.add_artist(circle)
            ax2.set_xlim(-1.2, 1.2)
            ax2.set_ylim(-1.2, 1.2)
            ax2.axhline(0, color='black', lw=1)
            ax2.axvline(0, color='black', lw=1)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
        st.info("💡 نصيحة: الدائرة الحمراء تظهر كيف ترتبط الأسئلة ببعضها. الأسهم المتقاربة تعني سلوكاً متشابهاً.")
    else:
        st.error("❌ خطأ: الملف يجب أن يحتوي على أرقام فقط (1-5) وأكثر من طالبين.")
