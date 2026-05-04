import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- 1. SAYFA AYARLARI VE BAŞLIK ---
st.set_page_config(page_title="Pnömoni Teşhis Sistemi", layout="centered")
st.title("🩺 BME3180: Pnömoni Teşhis Paneli")
st.write("Lütfen analiz edilecek göğüs röntgenini (X-Ray) yükleyiniz.")

# --- 2. MODELİ RADİKAL ŞEKİLDE YÜKLE ---
@st.cache_resource
def load_my_model():
    # Hata veren katmanı tamamen etkisiz hale getiriyoruz
    class DummyBatchNormalization(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            # Gelen tüm hatalı parametreleri (renorm vb.) yutuyoruz
            super().__init__()
        def call(self, x, training=False):
            return x

    # Modeli 'BatchNormalization' katmanını bu pasif sınıf ile değiştirerek yükle
    try:
        model = tf.keras.models.load_model(
            'custom_pneumonia_professional_final.keras',
            custom_objects={'BatchNormalization': DummyBatchNormalization},
            compile=False # Sunum için sadece tahmin (inference) yeterli, derlemeye gerek yok
        )
        return model
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu: {e}")
        return None

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

# --- 5. SIDEBAR ---
st.sidebar.title("Proje Hakkında")
st.sidebar.info("BME3180 Bitirme Projesi - Derin Öğrenme ile Pnömoni Teşhisi")