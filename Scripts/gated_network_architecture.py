# # **Gated Network Architecture**

# Define the GatedCombination class for combining two pairs of embeddings using a gating mechanism
class GatedCombination(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize the GatedCombination model.

        Args:
            input_dim (int): The dimensionality of the input embeddings (x1, x2, x3, x4).
        """
        super(GatedCombination, self).__init__()

        # Define a linear layer (gate) for combining embeddings x1 and x2 (first pair)
        self.gate_A_fc = nn.Linear(input_dim, input_dim)

        # Define a linear layer (gate) for combining embeddings x3 and x4 (second pair)
        self.gate_B_fc = nn.Linear(input_dim, input_dim)

        # A final fully connected layer that outputs a single neuron (binary classification)
        self.fc = nn.Linear(1, 1)

    def forward(self, x1, x2, x3, x4):
        """
        Forward pass through the gating mechanism and cosine similarity.

        Args:
            x1 (torch.Tensor): First set of embeddings (source embeddings after update).
            x2 (torch.Tensor): Second set of embeddings (original source embeddings).
            x3 (torch.Tensor): Third set of embeddings (target embeddings after update).
            x4 (torch.Tensor): Fourth set of embeddings (original target embeddings).

        Returns:
            torch.Tensor: Output of the model (probability score for binary classification).
        """
        # Compute gate values for the first pair (x1 and x2) using a sigmoid activation
        gate_values1 = torch.sigmoid(self.gate_A_fc(x1))

        # Combine x1 and x2 using the gate values
        # The result is a weighted combination of x1 and x2
        a = x1 * gate_values1 + x2 * (1 - gate_values1)

        # Compute gate values for the second pair (x3 and x4) using a sigmoid activation
        gate_values2 = torch.sigmoid(self.gate_B_fc(x3))

        # Combine x3 and x4 using the gate values
        # The result is a weighted combination of x3 and x4
        b = x3 * gate_values2 + x4 * (1 - gate_values2)

        # Compute cosine similarity between the combined vectors a and b
        x = torch.cosine_similarity(a, b, dim=1)

        # Pass the cosine similarity result through a fully connected layer (fc) for classification
        # Use a sigmoid activation to output a probability for binary classification
        out = torch.sigmoid(self.fc(x.unsqueeze(1)))  # unsqueeze(1) to match the input shape for the fc layer
        return out
