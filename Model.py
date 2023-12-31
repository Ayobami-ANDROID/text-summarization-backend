import pandas as pd
from transformers import AutoTokenizer
from transformers import pipeline
from flask import Flask, request, jsonify
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


summarize = pipeline("summarization",min_length=1,max_length=40)





class Summarizer:  # Renamed the class to 'Summarizer' (convention: use CamelCase for class names)
    def __init__(self):
        # Initialize your summarizer model or required variables here
        pass

    def generate_summary(self, sentences):
        summary = summarize(sentences)  # Assuming 'summarizer' is a function or method for summarization
        return(summary[0]['summary_text'].strip())


# Create an instance of your Summarizer class
my_summarizer = Summarizer()





# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Define the model with 3 output classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load the trained weights into the model
model.load_state_dict(torch.load('./classification_model.pth', map_location=torch.device('cpu')))
model.eval()

# Assuming 'model' and 'tokenizer' are defined and loaded

# Move model to CPU explicitly
device = torch.device("cpu")
model.to(device)

def predict(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move inputs to CPU
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        class_mapping = {
            0: 'dairy-farm_ar2020pdf',
            1: 'shell-tax-contribution-report-2020pdf',
            2 : 'with-highlights-comments-barclays-country-snapshot-2021pdf'
        }

        if predicted_class in class_mapping:
            return class_mapping[predicted_class]
        else:
            return 'Unknown Document'  # Return a default value for unknown classes


print(predict('Shell has been present in Pakistan since 1947'))

# Create an instance of your Summarizer class
@app.route('/v1/summarize', methods=['POST'])
def get_summary():
    try:
        data = request.get_json()
        if 'sentences' not in data:
            return jsonify({'error': 'Invalid request format. Missing "sentences" key.'}), 400
        
        sentences = data['sentences']
        summary = my_summarizer.generate_summary(sentences)
        
        return jsonify({'summary_text': summary})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def classify_document():
    try:
        data = request.get_json(force=True)
        if 'text' not in data:
            return jsonify({'error': 'Invalid request format. Missing "text" key.'}), 400
        
        text = data['text']
        result = predict(text)
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(port=7000, debug=True)

