import torch
import torch.nn as nn

class DualStreamEntropyClockNet(nn.Module):
    def __init__(self, input_dim):
        # We now accept input_dim so the model scales to any dataset automatically
        super(DualStreamEntropyClockNet, self).__init__()
        
        # 1. Methylation Stream
        self.meth_stream = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 2. Entropy Stream
        self.ent_stream = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 3. Late-Fusion Gated Attention Mechanism
        # Learns which stream to trust more based on the patient
        self.attention = nn.Sequential(
            nn.Linear(256, 128), # 128 from meth + 128 from ent
            nn.Tanh(),
            nn.Linear(128, 2),   # 2 weights: one for meth, one for ent
            nn.Softmax(dim=1)
        )
        
        # 4. Final Regressor
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, meth, ent):
        # Pass data through individual biological streams
        m_out = self.meth_stream(meth)
        e_out = self.ent_stream(ent)
        
        # Calculate attention weights dynamically
        combined = torch.cat([m_out, e_out], dim=1)
        attn_weights = self.attention(combined)
        
        # Split weights for each stream
        w_m = attn_weights[:, 0].unsqueeze(1)
        w_e = attn_weights[:, 1].unsqueeze(1)
        
        # Late Fusion: Weighted sum of both streams based on attention
        fused = (m_out * w_m) + (e_out * w_e)
        
        # Predict Chronological Age
        age_pred = self.regressor(fused)
        return age_pred