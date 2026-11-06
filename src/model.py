import torch
import torch.nn as nn

class DualStreamEntropyClockNet(nn.Module):
    def __init__(self, input_dim):

        super(DualStreamEntropyClockNet, self).__init__()

        # Two encoders with identical shape but separate weights. Keeping them
        # separate is the whole point: if meth and entropy were concatenated at
        # the input there'd be no boundary left to measure the split at.
        self.meth_stream = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.ent_stream = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Gate: takes both embeddings, returns one weight per stream.
        # Softmax so the two sum to 1 and can be read as percentages.
        self.attention = nn.Sequential(
            nn.Linear(256, 128),  # 128 meth + 128 ent
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, meth, ent):
        m_out = self.meth_stream(meth)
        e_out = self.ent_stream(ent)

        # Weights are computed per sample, so the split can differ between people
        combined = torch.cat([m_out, e_out], dim=1)
        attn_weights = self.attention(combined)

        w_m = attn_weights[:, 0].unsqueeze(1)
        w_e = attn_weights[:, 1].unsqueeze(1)

        # Convex combination, not concatenation -- keeps the output 128-dim
        fused = (m_out * w_m) + (e_out * w_e)

        # Output is standardised age; caller inverts the scaler to get years
        age_pred = self.regressor(fused)
        return age_pred
