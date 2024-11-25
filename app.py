
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "saidonepudi8/abstract_classification"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

labels = {
    0: "Neoplasms",
    1: "Digestive System Diseases",
    2: "Nervous System Diseases",
    3: "Cardiovascular Diseases"
}

def classify_abstract(abstract):
    inputs = tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    confidence = probabilities[0, predicted_class].item()
    
    # Fixed the unterminated f-string
    return f"Predicted Label: {labels[predicted_class]} Confidence: {confidence:.2f}"

st.title("Medical Abstract Classifier")
st.write("Enter a medical abstract and classify it into one of the disease categories.")

abstract_input = st.text_area("Enter Medical Abstract", height=300)

if st.button("Classify"):
    if abstract_input:
        result = classify_abstract(abstract_input)
        st.success(result)
    else:
        st.error("Please enter a medical abstract.")
    