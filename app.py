from flask import Flask, request, render_template, jsonify
from inference import *

app = Flask(__name__)

# Route for serving the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # This will serve index.html when the user navigates to the site

# Route for processing input from the client
@app.route('/process', methods=['POST'])
def process():
    user_input = request.form.get('message')  # Get the message from the form input
    if user_input:

        result, relevant_id, suggestion_ids  = generate(user_input)
        return jsonify({'result': result, 'productID':f'product-{relevant_id+1}'})  # Send back the computed result as JSON
    else:
        return jsonify({'error': 'No input provided'})

if __name__ == '__main__':
    app.run(debug=True)