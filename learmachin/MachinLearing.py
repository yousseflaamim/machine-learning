from flask import Flask, request
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.out(x)
        return x


input_size = 1
hidden_size = 32
output_size = 1


model = Net(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("trained_model1.pth"))


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
   
    data = request.json
    
    data_list = list(data.values())
    x = torch.tensor(data_list, dtype=torch.float32).view(-1, input_size)

 
    output = model(x)
    
    result = str(output.detach().numpy()[0][0])
   
    return result

if __name__ == '__main__':
    app.run(debug=True,port=5555)