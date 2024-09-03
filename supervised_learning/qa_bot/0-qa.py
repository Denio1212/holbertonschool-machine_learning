import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def question_answering(question, context):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained("deepset/coqa-transformer-base-v1", num_labels=2)

    # Tokenize the question and context
    inputs = tokenizer(question, context, return_tensors="pt")

    # Pass the tokenized input to the model
    outputs = model(**inputs)

    # Get the start and end positions of the answer from the output
    start_position = torch.argmax(outputs[0][0]) + 1
    end_position = torch.argmax(outputs[1][0]) + 1

    # Extract the answer text from the context using the start and end positions
    answer_start = context.find(question) + len(question)
    answer_end = min(
        answer_start + (context[answer_start:] if isinstance(context, str) else context[answer_start])).find(".") + 1
    answer = context[answer_start:answer_end].strip()

    return answer



