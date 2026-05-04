import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- 1. SAYFA AYARLARI VE BAŞLIK ---
st.set_page_config(page_title="Pnömoni Teşhis Sistemi", layout="centered")
st.title("🩺 BME3180: Pnömoni Teşhis Paneli")
st.write("Lütfen analiz edilecek göğüs röntgenini (X-Ray) yükleyiniz.")

# --- 2. MODELİ YÜKLE ---
@st.cache_resource # Modelin her seferinde tekrar yüklenip kasmasını önler
def load_my_model():
    # Kaydettiğin modelin isminin aynı olduğundan emin ol
    model = tf.keras.models.load_model('custom_pneumonia_professional_final.keras')
    return model

with st.spinner('Model yükleniyor, lütfen bekleyiniz...'):
    model = load_my_model()

# --- 3. DOSYA YÜKLEME ALANI ---
file = st.file_uploader("Görüntü Seçiniz", type=["jpg", "png", "jpeg"])

def predict_pneumonia(image_data, model):
    # Görüntüyü modelin eğitildiği boyuta (150x150) getiriyoruz
    size = (150, 150)    
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalizasyon ve Boyut Ayarlama (Batch boyutu ekleme)[cite: 1]
    img_reshape = img_array.astype('float32') / 255.0
    img_reshape = np.expand_dims(img_reshape, axis=0) # (1, 150, 150, 3)
    
    # Tahmin
    prediction = model.predict(img_reshape)
    return prediction

# --- 4. SONUÇLARI GÖSTER ---
if file is None:
    st.text("Henüz bir görüntü yüklenmedi.")
else:
    # Yüklenen görüntüyü ekranda göster
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Yüklenen Röntgen Görüntüsü', use_container_width=True)
    
    # Analiz butonu
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
            
            st.info("Not: Bu sonuç bir yapay zeka tahminidir. Kesin teşhis için uzman bir radyolog onayı gereklidir.")

# --- 5. AKADEMİK BİLGİ ---
st.sidebar.title("Proje Hakkında")
st.sidebar.info(
    "Bu model, özgün bir CNN mimarisi kullanılarak "
    "1137 test görseli üzerinde %87 doğruluk ve "
    "0.9279 AUC skoru ile eğitilmiştir.[cite: 1]"
)