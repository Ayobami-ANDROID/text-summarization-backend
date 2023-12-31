# Document Summarization and Classification API

## Overview

This API provides functionalities for document summarization and classification using Natural Language Processing (NLP) techniques.

### Technologies Used

- Python
- Flask
- Hugging Face Transformers
- PyTorch

## Features

### Summarization Functionality

The API offers a `/v1/summarize` endpoint that generates a concise summary of a given text or a set of sentences. It utilizes the `transformers.pipeline` for text summarization.

#### Endpoint Details

- **Endpoint:** `/v1/summarize`
- **Method:** POST
- **Input:** JSON object with a key "sentences" containing a list of sentences
- **Output:** JSON object with a summarized text under the key "summary_text"

### Document Classification Functionality

The API provides a `/predict` endpoint for classifying documents into predefined categories. It uses a pre-trained BERT-based model for document classification.

#### Endpoint Details

- **Endpoint:** `/predict`
- **Method:** POST
- **Input:** JSON object with a key "text" containing the document text
- **Output:** JSON object with the predicted class label under the key "prediction"

## Data Engineering/Processing

The API processes raw text data by tokenizing and encoding it using Transformers' tokenizer before feeding it to the models for summarization and classification.

## Challenges Faced

One of the main challenges was optimizing the model inference time, especially for large documents, to ensure real-time response. This was addressed by batching and optimizing the input data.

## Future Improvements

Given more time, the backend of the API could be improved in several ways:
- Implementation of caching mechanisms for frequently requested data.
- Fine-tuning the summarization and classification models for better accuracy.
- Scaling the API to handle higher concurrent requests by deploying it on a cloud platform.
