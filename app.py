from flask import Flask, request, render_template, jsonify
from faker import Faker
import pickle
import json

# Initialize Flask app and Faker instance
app = Flask(__name__)
faker = Faker()

# Load the model and vectorizer (replace with correct paths)
with open('ml/chatbot_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('ml/cyber.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# User Query Handling & Error Guidance (Chatbot logic)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    processed_input = vectorizer.transform([user_input])
    
    # Predict the response using the model
    bot_response = model.predict(processed_input)[0]
    
    return render_template('index.html', user_input=user_input, bot_response=bot_response)

# Contextual Information Retrieval (Sample Data Simulation)
@app.route('/context_info', methods=['GET'])
def context_info():
    # Simulated retrieval of website mind map, page states, etc.
    context_data = {
        'current_page': 'HomePage',
        'test_status': 'Running',
        'navigation_path': ['HomePage', 'LoginPage', 'Dashboard']
    }
    return jsonify(context_data)

# Mock Data & Content Generation (Faker Integration)
@app.route('/mock_data', methods=['GET'])
def generate_mock_data():
    mock_data = {
        'username': faker.user_name(),
        'email': faker.email(),
        'password': faker.password(),
        'address': faker.address()
    }
    return jsonify(mock_data)

# Dynamic Instruction & Process Control (Command Execution Simulation)
@app.route('/execute_command', methods=['POST'])
def execute_command():
    command = request.json.get('command')
    
    # Example command execution logic
    if command == 'run_test':
        response = {'status': 'Test execution started'}
    elif command == 'stop_test':
        response = {'status': 'Test execution stopped'}
    else:
        response = {'status': f'Unknown command: {command}'}
    
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
