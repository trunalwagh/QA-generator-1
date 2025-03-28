from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import random
import torch

app = Flask(__name__)

# Initialize models and tokenizers
t5_model_name = 'valhalla/t5-base-qg-hl'
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

# Ensure the correct device is used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_model.to(device)

qa_pipeline = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

def generate_questions(paragraph, num_questions=5):
    text = f"generate questions: {paragraph}"
    inputs = t5_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(device)

    outputs = t5_model.generate(
        inputs,
        max_length=500,
        num_beams=10,
        num_return_sequences=num_questions,
        early_stopping=True
    )

    questions = [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

def extract_possible_answers(paragraph):
    sentences = paragraph.split('. ')
    possible_answers = [sentence.strip() for sentence in sentences if len(sentence.split()) <= 3]
    return possible_answers

def generate_mcqs(paragraph, questions):
    possible_answers = extract_possible_answers(paragraph)
    
    mcqs = []
    for question in questions:
        result = qa_pipeline(question=question, context=paragraph)
        correct_answer = result['answer']
        
        options = [correct_answer]
        if correct_answer in possible_answers:
            possible_answers.remove(correct_answer)
        
        while len(options) < 4 and possible_answers:
            wrong_answer = random.choice(possible_answers)
            if wrong_answer and wrong_answer not in options:
                options.append(wrong_answer)
                possible_answers.remove(wrong_answer)
        
        while len(options) < 4:
            options.append("N/A")
        
        random.shuffle(options)
        
        mcqs.append({
            'question': question,
            'options': options,
            'correct_answer': correct_answer
        })
    return mcqs

def extract_answers(paragraph, questions):
    answers = []
    for question in questions:
        result = qa_pipeline(question=question, context=paragraph)
        answers.append(result['answer'])
    return answers

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        paragraph = request.form['paragraph']
        generate_type = request.form['generate_type']
        num_questions = int(request.form['num_questions'])

        questions = generate_questions(paragraph, num_questions=num_questions)

        output = {}
        if generate_type == '1':
            mcqs = generate_mcqs(paragraph, questions)
            output['mcqs'] = mcqs
        elif generate_type == '2':
            answers = extract_answers(paragraph, questions)
            output['qa'] = list(zip(questions, answers))
        elif generate_type == '3':
            mcqs = generate_mcqs(paragraph, questions)
            output['mcqs'] = mcqs
            answers = extract_answers(paragraph, questions)
            output['qa'] = list(zip(questions, answers))

        return render_template('output.html', output=output)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
