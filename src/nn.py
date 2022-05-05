import torch
import torch.nn.functional as F

class PolicyNN(torch.nn.Module):

    def __init__(self, n_feature, n_layers, n_hidden : int, n_output : int):
        '''
        n_feature = number of inputs
        n_hidden = number of nodes in hidden layer
        n_layers = number of hidden layers
        n_output = number of outputs
        '''
        super(PolicyNN, self).__init__()
        self.hidden0 = (torch.nn.Linear(n_feature, n_hidden))
        self.hidden1 = (torch.nn.Linear(n_hidden, n_hidden))
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, input):
        layer_output = F.relu(self.hidden0(input))
        layer_output = F.relu(self.hidden1(layer_output))      # activation function for hidden layer
        actions = self.predict(layer_output)
        network_output = F.softmax(actions)    # linear output
        return network_output

class ValueNN(torch.nn.Module):
    def __init__(self, n_feature, n_layers, n_hidden : int, n_output):
        super(ValueNN, self).__init__()
        self.hidden0 = (torch.nn.Linear(n_feature, n_hidden))
        self.hidden1 = (torch.nn.Linear(n_hidden, n_hidden))
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, input):
        layer_output = F.relu(self.hidden0(input))
        layer_output = F.relu(self.hidden1(layer_output))      # activation function for hidden layer
        network_output = self.predict(layer_output)    # linear output
        return network_output