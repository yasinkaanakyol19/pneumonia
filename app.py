import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Pnömoni Teşhis Sistemi", layout="centered")
st.title("🩺 BME3180: Pnömoni Teşhis Paneli")

# --- 2. MODELİ HATALARI AYIKLAYARAK YÜKLE ---
@st.cache_resource
def load_my_model():
    from keras.layers import BatchNormalization

    # Keras'ın model dosyasından okuduğu ama tanımadığı parametreleri 
    # süper sınıfa göndermeden önce ayıklayan özel sınıf.
    class SafeBatchNormalization(BatchNormalization):
        def __init__(self, **kwargs):
            # Hata veren tüm anahtarları kwargs sözlüğünden siliyoruz
            for key in ['renorm', 'renorm_clipping', 'renorm_momentum']:
                kwargs.pop(key, None)
            super().__init__(**kwargs)

    try:
        # Modeli bu özel 'Safe' katman ile yüklüyoruz. 
        # custom_objects sayesinde Keras hata vermek yerine bizim sınıfımızı kullanacak.
        model = tf.keras.models.load_model(
            'custom_pneumonia_professional_final.keras',
            custom_objects={'BatchNormalization': SafeBatchNormalization},
            compile=False # Tahmin için eğitim metriklerine gerek yok, hata payını azaltır
        )
        return model
    except Exception as e:
        st.error(f"Kritik Yükleme Hatası: {e}")
        return None

with st.spinner('Yapay Zeka motoru hazırlanıyor, lütfen bekleyiniz...'):
    model = load_my_model()

# --- 3. ARAYÜZ VE ANALİZ ---
file = st.file_uploader("Lütfen Röntgen Görselini Yükleyin", type=["jpg", "png", "jpeg"])

if file and model:
    # Görüntüyü göster
    img = Image.open(file).convert('RGB')
    st.image(img, caption='Analiz Edilecek Röntgen', use_container_width=True)
    
    if st.button("Teşhis Et"):
        with st.spinner('Yapay zeka analiz ediyor...'):
            # Görüntü hazırlama (150x150)
            size = (150, 150)
            processed_img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(processed_img).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 150, 150, 3)
            
            # Tahmin yürütme
            prediction = model.predict(img_array)
            score = float(prediction.flatten()[0])
            
            st.divider()
            if score > 0.5:
                st.error(f"🚨 SONUÇ: PNÖMONİ (ZATÜRRE) BELİRTİLERİ SAPTANDI")
                st.write(f"**Güven Oranı:** %{score * 100:.2f}")
            else:
                st.success(f"✅ SONUÇ: AKCİĞERLER NORMAL GÖRÜNÜYOR")
                st.write(f"**Güven Oranı:** %{(1 - score) * 100:.2f}")
            
            st.info("Bilgi: Bu araç Biyomedikal Mühendisliği bitirme projesi kapsamında geliştirilmiştir.")