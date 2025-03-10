import torch
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout, CosineSimilarity, Module

class EdgeFeatureParser(Module):
    def __init__(self, input_dim, dropout=0.2):
        super(EdgeFeatureParser, self).__init__()
        hidden_dim = 128
        self.input_dim = input_dim

        self.parser = Sequential(
            Linear(input_dim, hidden_dim, bias=True),
            ReLU(inplace=True),
            BatchNorm1d(hidden_dim),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim, bias=True)
        )

        self.cosine_similarity = CosineSimilarity(dim=1, eps=1e-8)
        self._initialize_model_weights()

    def forward(self, x):

        features1 = x[:, 0:self.input_dim]
        features2 = x[:, self.input_dim:2 * self.input_dim]
        parsed_features1, parsed_features2 = self.parser(features1), self.parser(features2)
        similarity_scores = (self.cosine_similarity(parsed_features1, parsed_features2) + 1) * 0.5
        return similarity_scores

    def _initialize_model_weights(self):

        for module in self.modules():
            if isinstance(module, Linear):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
