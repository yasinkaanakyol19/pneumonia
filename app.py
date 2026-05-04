import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Pnömoni Teşhis Sistemi", layout="centered")
st.title("🩺 BME3180: Pnömoni Teşhis Paneli")

# --- 2. MODELİ TÜM HATALARI BYPASS EDEREK YÜKLE ---
@st.cache_resource
def load_my_model():
    # BatchNormalization'daki 'renorm' hatasını aşmak için en basit yöntem: 
    # Katmanı özel bir nesne olarak değil, Keras'ın kendi sınıfını 
    # ama hatalı argümanları yutacak şekilde kandırarak yüklüyoruz.
    
    from keras.layers import BatchNormalization

    class CustomBatchNormalization(BatchNormalization):
        def __init__(self, **kwargs):
            # Model dosyasından gelen ama bu versiyonda olmayan tüm parametreleri siliyoruz
            bad_keys = ['renorm', 'renorm_clipping', 'renorm_momentum']
            for key in bad_keys:
                kwargs.pop(key, None)
            super().__init__(**kwargs)

    try:
        # compile=False yaparak eğitim metriklerinin yüklenmesini pas geçiyoruz (tahmin için gerek yok)
        model = tf.keras.models.load_model(
            'custom_pneumonia_professional_final.keras',
            custom_objects={'BatchNormalization': CustomBatchNormalization},
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Kritik Yükleme Hatası: {e}")
        return None

with st.spinner('Model ve Yapay Zeka motoru hazırlanıyor...'):
    model = load_my_model()

# --- 3. ARAYÜZ VE TAHMİN ---
file = st.file_uploader("Röntgen Görseli Yükle", type=["jpg", "png", "jpeg"])

if file and model:
    img = Image.open(file).convert('RGB')
    st.image(img, caption='Analiz Edilecek Görüntü', use_container_width=True)
    
    if st.button("Analizi Başlat"):
        # Görüntü işleme (Eğitimdeki gibi 150x150)
        size = (150, 150)
        processed_img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(processed_img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        score = float(prediction.flatten()[0])
        
        st.divider()
        if score > 0.5:
            st.error(f"🚨 SONUÇ: PNÖMONİ BELİRTİLERİ SAPTANDI (Güven: %{score*100:.2f})")
        else:
            st.success(f"✅ SONUÇ: AKCİĞERLER NORMAL GÖRÜNÜYOR (Güven: %{(1-score)*100:.2f})")