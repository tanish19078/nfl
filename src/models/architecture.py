import torch
import torch.nn as nn
import torch.nn.functional as F

class PlayerLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(PlayerLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        # x: (Batch, SeqLen, InputDim)
        # output: (Batch, SeqLen, HiddenDim)
        output, (hn, cn) = self.lstm(x)
        return output, (hn, cn)

class InteractionGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InteractionGNN, self).__init__()
        # Simple Graph Attention Layer equivalent
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        # x: (Batch, NumPlayers, HiddenDim)
        # Compute attention scores between all pairs of players
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores: (Batch, NumPlayers, NumPlayers)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of values
        context = torch.matmul(attn_weights, V)
        return context

class TrajectoryDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(TrajectoryDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden):
        # x: (Batch, 1, InputDim) - current step input
        # hidden: (hn, cn) from encoder or previous step
        output, (hn, cn) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hn, cn)

class NFLPredictor(nn.Module):
    def __init__(self, input_dim=33, hidden_dim=64, output_dim=2, pred_len=25):
        super(NFLPredictor, self).__init__()
        self.pred_len = pred_len
        self.encoder = PlayerLSTMEncoder(input_dim, hidden_dim)
        self.gnn = InteractionGNN(hidden_dim, hidden_dim)
        self.decoder = TrajectoryDecoder(output_dim, hidden_dim, output_dim) # Input to decoder is prev pos (2 dim)
        
        # Project GNN output back to hidden dim for decoder init if needed, 
        # or just use encoder hidden state enhanced by GNN.
        # Here we'll use GNN to update the hidden state before decoding.
        self.gnn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (Batch, NumPlayers, HistoryLen, InputDim)
        batch_size, num_players, history_len, input_dim = x.shape
        
        # Flatten batch and players for LSTM: (Batch*NumPlayers, HistoryLen, InputDim)
        x_flat = x.view(batch_size * num_players, history_len, input_dim)
        
        # Encode history
        enc_out, (hn, cn) = self.encoder(x_flat)
        # hn: (NumLayers, Batch*NumPlayers, HiddenDim)
        
        # Reshape for GNN: (Batch, NumPlayers, HiddenDim)
        # Use last hidden state
        last_hidden = hn[-1].view(batch_size, num_players, -1)
        
        # Apply GNN to capture interactions at the last timestep of history
        gnn_out = self.gnn(last_hidden)
        # Residual connection + Norm
        interaction_aware_hidden = self.gnn_norm(last_hidden + gnn_out)
        
        # Prepare for decoder
        # Reshape back to (1, Batch*NumPlayers, HiddenDim) for LSTM
        decoder_hidden = (interaction_aware_hidden.view(1, batch_size * num_players, -1), cn)
        
        # Decoding loop
        predictions = []
        # Start with the last known position (x, y) from input
        # Assuming x, y are the first 2 features
        current_input = x_flat[:, -1, :2].unsqueeze(1) # (Batch*NumPlayers, 1, 2)
        
        for _ in range(self.pred_len):
            pred, decoder_hidden = self.decoder(current_input, decoder_hidden)
            predictions.append(pred)
            current_input = pred # Autoregressive
            
        # Stack predictions: (Batch*NumPlayers, PredLen, 2)
        predictions = torch.cat(predictions, dim=1)
        
        # Reshape back: (Batch, NumPlayers, PredLen, 2)
        predictions = predictions.view(batch_size, num_players, self.pred_len, 2)
        
        return predictions
