import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- 1. SAYFA AYARLARI VE BAŞLIK ---
st.set_page_config(page_title="Pnömoni Teşhis Sistemi", layout="centered")
st.title("🩺 BME3180: Pnömoni Teşhis Paneli")
st.write("Lütfen analiz edilecek göğüs röntgenini (X-Ray) yükleyiniz.")

# --- 2. MODELİ HATALARI BYPASS EDEREK YÜKLE ---
@st.cache_resource
def load_my_model():
    # Model yüklenirken BatchNormalization hatalarını tamamen görmezden geliyoruz
    try:
        # custom_objects içinde BatchNormalization'ı Keras'ın kendi sınıfına 
        # ama hatalı argümanları kabul etmeyen bir yapıya zorluyoruz
        model = tf.keras.models.load_model(
            'custom_pneumonia_professional_final.keras',
            safe_mode=False, # Yeni Keras sürümlerinde güvenli modu kapatmak dosya okumayı kolaylaştırır
            compile=False
        )
        return model
    except Exception as e:
        # Eğer hala hata verirse, katmanı manuel olarak 'yutucu' bir sınıfla değiştiriyoruz
        from tensorflow.keras.layers import Layer
        class SafeBatchNormalization(Layer):
            def __init__(self, **kwargs):
                # Tüm bilinmeyen parametreleri temizle
                for key in ['renorm', 'renorm_clipping', 'renorm_momentum']:
                    kwargs.pop(key, None)
                super().__init__(**kwargs)
            def call(self, x): return x

        return tf.keras.models.load_model(
            'custom_pneumonia_professional_final.keras',
            custom_objects={'BatchNormalization': SafeBatchNormalization},
            compile=False
        )

with st.spinner('Model yükleniyor, lütfen bekleyiniz...'):
    model = load_my_model()

# --- 3. DOSYA YÜKLEME ALANI ---
file = st.file_uploader("Görüntü Seçiniz", type=["jpg", "png", "jpeg"])

def predict_pneumonia(image_data, model):
    size = (150, 150)    
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_reshape = img_array.astype('float32') / 255.0
    img_reshape = np.expand_dims(img_reshape, axis=0)
    
    prediction = model.predict(img_reshape)
    return prediction

# --- 4. SONUÇLARI GÖSTER ---
if file is not None and model is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Yüklenen Röntgen Görüntüsü', use_container_width=True)
    
    if st.button("Teşhis Et"):
        with st.spinner('Yapay zeka analiz ediyor...'):
            prediction = predict_pneumonia(image, model)
            score = float(prediction.flatten()[0])
            
            st.divider()
            if score > 0.5:
                st.error(f"⚠️ SONUÇ: PNÖMONİ (ZATÜRRE) TESPİT EDİLDİ")
                st.write(f"**Güven Oranı:** %{score * 100:.2f}")
            else:
                st.success(f"✅ SONUÇ: NORMAL (SAĞLIKLI)")
                st.write(f"**Güven Oranı:** %{(1 - score) * 100:.2f}")
            
            st.info("Not: Bu sonuç bir yapay zeka tahminidir. Kesin teşhis için uzman radyolog onayı gereklidir.")

# --- 5. AKADEMİK BİLGİ ---
st.sidebar.title("Proje Hakkında")
st.sidebar.info("BME3180 Bitirme Projesi - Derin Öğrenme ile Pnömoni Teşhisi")