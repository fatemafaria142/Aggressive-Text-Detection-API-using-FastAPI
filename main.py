from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np


# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model and tokenizer
num_classes = 4
model_name = "csebuetnlp/banglabert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("Soyeda10/AggressiveBanglaBERT", num_labels=num_classes)

# Dictionary to map predicted labels to their corresponding categories
dictionary_labels = {0: 'Religious Aggression', 1: 'Political Aggression', 2: 'Verbal Aggression', 3: 'Gendered Aggression'}

# Request body model
class TextInput(BaseModel):
    text: str

# Define prediction route
@app.post("/predict/")
async def predict_text(text_input: TextInput):
    text = text_input.text
    test_sample = tokenizer([text], padding=True, truncation=True, max_length=250, return_tensors='pt')
    output = model(**test_sample)
    y_predicted = np.argmax(output.logits.detach().numpy(), axis=1)
    predicted_label = dictionary_labels[y_predicted[0]]
    return {"prediction": predicted_label}