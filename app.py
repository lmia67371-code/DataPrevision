import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("تحليل المكونات الأساسية (PCA)")

uploaded_file = st.file_uploader("ارفع ملف البيانات (CSV)", type="csv")

if uploaded_file:
    # قراءة الملف بالفاصلة المنقوطة كما اتفقنا
    df = pd.read_csv(uploaded_file, sep=';')
    st.write("تم استقبال البيانات بنجاح ✅")
    
    # اختيار الأعمدة الرقمية فقط
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.shape[0] > 1:
        # معالجة البيانات
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # تطبيق PCA
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)
        
        # الرسم البياني للأفراد (الطلاب)
        st.subheader("خريطة الطلاب (Individuals Map)")
        fig1, ax1 = plt.subplots()
        plt.scatter(components[:, 0], components[:, 1], c='blue', alpha=0.7)
        plt.xlabel("المحور 1")
        plt.ylabel("المحور 2")
        st.pyplot(fig1)
        
        # دائرة الارتباطات (الأسئلة)
        st.subheader("دائرة الارتباطات (Variables Circle)")
        fig2, ax2 = plt.subplots()
        for i, var in enumerate(numeric_df.columns):
            plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], head_width=0.05, head_length=0.05)
            plt.text(pca.components_[0, i]*1.2, pca.components_[1, i]*1.2, var, color='red')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        st.pyplot(fig2)
    else:
        st.error("يرجى رفع ملف يحتوي على أكثر من طالب واحد.")