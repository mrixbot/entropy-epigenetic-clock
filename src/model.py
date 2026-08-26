import torch
import torch.nn as nn

class DualStreamEntropyClockNet(nn.Module):
    def __init__(self, n_features_per_stream, embed_dim=64, dropout=0.3):
        super().__init__()
        self.n_features_per_stream = n_features_per_stream

        # Branch 1: Methylation Encoder
        self.methylation_branch = nn.Sequential(
            nn.Linear(n_features_per_stream, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LeakyReLU(0.2),
        )

        # Branch 2: Entropy Encoder
        self.entropy_branch = nn.Sequential(
            nn.Linear(n_features_per_stream, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LeakyReLU(0.2),
        )

        # Gated Fusion Layer
        # Takes the concatenated embeddings and calculates a fusion gate (0 to 1)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        # Final Prediction Head
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # 1. Split the inputs perfectly in half
        meth = x[:, : self.n_features_per_stream]
        entropy = x[:, self.n_features_per_stream :]

        # 2. Get embeddings from both branches
        meth_embed = self.methylation_branch(meth)
        entropy_embed = self.entropy_branch(entropy)

        # 3. Calculate the Gate (z)
        # We concatenate them just to calculate the gating score
        concat_embeds = torch.cat([meth_embed, entropy_embed], dim=1)
        z = self.fusion_gate(concat_embeds)

        # 4. Gated Fusion
        # If z is close to 1, it favors methylation. If z is close to 0, it favors entropy.
        fused_embed = (z * meth_embed) + ((1 - z) * entropy_embed)

        # 5. Predict Age
        return self.prediction_head(fused_embed)