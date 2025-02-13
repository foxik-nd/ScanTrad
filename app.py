import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------- CONFIG -------------------
YOLO_MODEL_PATH = "weights/best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Charger le modèle YOLOv5
@st.cache_resource
def load_yolo():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=True)
    return model

# Charger les modèles de traduction et de détection de langue
@st.cache_resource
def load_translation_models():
    lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    return lang_detector, tokenizer, model 


# Détection des bulles et extraction du texte

def extract_text_from_bubbles(image, bubbles):
    extracted_texts = []
    for (x1, y1, x2, y2) in bubbles:
        roi = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(roi, lang="eng").strip()
        extracted_texts.append((text, (x1, y1, x2, y2)))
    return extracted_texts

# Détection automatique de la langue
def detect_language(text, lang_detector):
    if text.strip():
        result = lang_detector(text)
        return result[0]['label']
    return "unknown"

# Traduction avec NLLB-200
def translate_text(text, tokenizer, model, target_language="fra_Latn"):
    inputs = tokenizer(text, return_tensors="pt")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_language)
    outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Dessiner les bulles traduites sur l'image
def draw_translations(image, texts_translated):
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()
    for (text, (x1, y1, x2, y2)) in texts_translated:
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text((x1, y1 - 15), text, fill="red", font=font)
    return np.array(image_pil)

# ------------------- APPLICATION STREAMLIT -------------------
st.title("📖 ScanTrad IPSSI 📖")
st.write("Cette application détecte les bulles de texte, les traduit automatiquement en français et affiche le résultat.")
st.write("Elle a été développée par : Nassim YAZI, Sokpagna HEANG, Nawfel ZENZELAOUI, Nicolas DESFORGES.")

uploaded_file = st.file_uploader("📤 Uploade une page de manga en anglais", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Image originale", use_column_width=True)

    model = load_yolo()
    lang_detector, tokenizer, translation_model = load_translation_models()
    
    results = model(image_np)
    detections = results.xyxy[0].cpu().numpy()
    bubbles = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2, conf, cls in detections]

    if bubbles:
        st.success(f"✅ {len(bubbles)} bulles détectées")
        extracted_texts = extract_text_from_bubbles(image_np, bubbles)
        texts = [text for text, _ in extracted_texts if text]

        if texts:
            st.write("🔠 **Texte original extrait** :")
            st.write("\n".join(texts))
            detected_langs = [detect_language(text, lang_detector) for text in texts]
            st.write("🌍 **Langues détectées :**", detected_langs)
            translated_texts = [translate_text(text, tokenizer, translation_model) for text in texts]
            st.write("🌐 **Traduction :**")
            st.write("\n".join(translated_texts))
            texts_translated = list(zip(translated_texts, [coords for _, coords in extracted_texts]))
            translated_image = draw_translations(image_np, texts_translated)
            st.image(translated_image, caption="Image traduite", use_column_width=True)
            translated_image_pil = Image.fromarray(translated_image)
            translated_image_pil.save("translated_image.png")
            with open("translated_image.png", "rb") as file:
                st.download_button(label="📥 Télécharger l'image traduite", data=file, file_name="translated_manga.png")
        else:
            st.error("⚠️ Aucun texte n'a été extrait de l'image.")
    else:
        st.error("⚠️ Aucune bulle détectée.")
