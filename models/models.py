import torch

class mlp(torch.nn.Module):
    def __init__(self,input_dimension,hidden_dimension,model_layers,
              activation, complex_valued = False, normalization_flag = False):
        super().__init__()
        self.model_layers = model_layers
        self.normalization_flag = normalization_flag

        if(activation == 'sine'):
            self.activation = torch.sin
        elif(activation == 'relu'):
            self.activation = torch.nn.functional.relu

        self.layers = torch.nn.ModuleList([])
        for ii in range(model_layers):
            if(ii == 0): # first layer goes from input dimension to hidden dimension
                self.layers.append(torch.nn.Linear(input_dimension,hidden_dimension))
            elif(ii < (model_layers-1)):
                self.layers.append(torch.nn.Linear(hidden_dimension,hidden_dimension))
            else: # last layer goes from hidden dimension to 1 (real) or 2 (complex)
                if complex_valued:
                    self.layers.append(torch.nn.Linear(hidden_dimension,2))
                else:
                    self.layers.append(torch.nn.Linear(hidden_dimension,1))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def normalization(self, inp):
        scale = (1/(torch.std(inp, correction=0, dim=-1, keepdim=True) + 1e-5))
        mean = torch.mean(inp, dim=-1, keepdim=True)

        return scale * (inp - mean)

    def forward(self,x):
        for ii,layer in enumerate(self.layers):
            x = layer(x)
            if ii < (self.model_layers-1): # perform activation (and normalization if desired) if not last layer
                x = self.activation(x)
                if self.normalization_flag: x = self.normalization(x)

        return x

