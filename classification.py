from flask import Flask, request, jsonify
import torch
from transformers import BertForSequenceClassification, BertTokenizer

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Define the model with 3 output classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load the trained weights into the model
model.load_state_dict(torch.load('./classification_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define a function for prediction
def predict(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        if predicted_class == 0:
            return 'dairy-farm_ar2020pdf'
        elif predicted_class ==1:
            return 'shell-tax-contribution-report-2020pdf'
        else:
            return 'with-highlights-comments-barclays-country-snapshot-2021pdf'
# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def prediction():
    data = request.get_json(force=True)
    text = data['text']
    result = predict(text)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
