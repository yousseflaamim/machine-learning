from flask import Flask, request, jsonify,json
import torch

app = Flask(__name__)

# Load the trainer model
model = torch.load('trained_model.pth')


@app.route('/predict', methods=['POST'])
def predict():
    
    input_data =json.loads( request.data) 
    print(input_data)
    input_tensor = torch.Tensor([input_data['input']])

    prediction = model(input_tensor).item()
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    
    app.run(debug=True )