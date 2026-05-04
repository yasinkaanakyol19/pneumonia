import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Pnömoni Teşhis Sistemi", layout="centered")
st.title("🩺 BME3180: Pnömoni Teşhis Paneli")

# --- 2. KESİN ÇÖZÜM: MODELİ MANUEL İNŞA ET VE AĞIRLIKLARI YÜKLE ---
@st.cache_resource
def load_my_model():
    # Hata veren Keras okuyucusunu atlıyor, model iskeletini kendimiz çiziyoruz
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(150, 150, 3)),
        
        # 1. Blok
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # 2. Blok
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # 3. Blok
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # 4. Blok
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Sınıflandırıcı
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Bozuk konfigürasyonları pas geçip, SADECE senin eğittiğin ağırlıkları (kasları) yüklüyoruz!
    model.load_weights('custom_pneumonia_professional_final.keras')
    return model

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
            img_array = np.expand_dims(img_array, axis=0)
            
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
            
            st.info("Bilgi: Bu araç eğitim amaçlı geliştirilmiştir. Kesin teşhis için uzman doktor onayı gereklidir.")