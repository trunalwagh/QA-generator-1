
from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import random
import torch

app = Flask(__name__)

# Initialize models and tokenizers
t5_model_name = 'valhalla/t5-base-qg-hl'
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, legacy=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_model.to(device)

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


def validate_input(paragraph):
    return paragraph and len(paragraph.split()) > 5 and any(c.isalpha() for c in paragraph)


def generate_questions(paragraph, num_questions=5):
    text = f"generate questions: {paragraph}"
    inputs = t5_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(device)

    outputs = t5_model.generate(
        inputs,
        max_length=128,
        num_beams=5,  # Reduced from 10 to optimize memory
        num_return_sequences=min(num_questions, 5),
        early_stopping=True
    )
    return [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


def extract_possible_answers(paragraph):
    sentences = paragraph.split('. ')
    return [s.strip() for s in sentences if 1 <= len(s.split()) <= 3]


def generate_mcqs(paragraph, questions):
    possible_answers = extract_possible_answers(paragraph)
    mcqs = []

    for question in questions:
        try:
            result = qa_pipeline(question=question, context=paragraph)
            correct_answer = result.get('answer', 'N/A')
        except:
            correct_answer = 'N/A'

        options = [correct_answer]
        if correct_answer in possible_answers:
            possible_answers.remove(correct_answer)

        while len(options) < 4 and possible_answers:
            wrong_answer = random.choice(possible_answers)
            if wrong_answer not in options:
                options.append(wrong_answer)
                possible_answers.remove(wrong_answer)

        while len(options) < 4:
            options.append("N/A")

        random.shuffle(options)
        mcqs.append({'question': question, 'options': options, 'correct_answer': correct_answer})

    return mcqs


def extract_answers(paragraph, questions):
    answers = []
    for question in questions:
        try:
            result = qa_pipeline(question=question, context=paragraph)
            answers.append(result.get('answer', 'N/A'))
        except:
            answers.append('N/A')
    return answers


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        paragraph = request.form['paragraph']
        generate_type = request.form['generate_type']
        num_questions = int(request.form['num_questions'])

        if not validate_input(paragraph):
            return render_template('index.html', error="Invalid input. Please enter meaningful text.")

        questions = generate_questions(paragraph, num_questions)
        output = {}

        if generate_type == '1':
            output['mcqs'] = generate_mcqs(paragraph, questions)
        elif generate_type == '2':
            output['qa'] = list(zip(questions, extract_answers(paragraph, questions)))
        elif generate_type == '3':
            output['mcqs'] = generate_mcqs(paragraph, questions)
            output['qa'] = list(zip(questions, extract_answers(paragraph, questions)))

        return render_template('output.html', output=output)

    return render_template('index.html')


# if __name__ == "__main__":
#     app.run(debug=True)
