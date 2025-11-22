import torch
import torch.nn as nn

class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=2, dropout=0.2):
        super(TrajectoryPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Layer to map hidden state to output coordinates
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, future_frames=0, teacher_forcing_ratio=0.0):
        """
        Args:
            x: Input sequence (Batch, Seq_Len, Features)
            future_frames: Number of future frames to predict autoregressively (optional)
            teacher_forcing_ratio: Not used in this simple version, but good for seq2seq
        Returns:
            predictions: (Batch, Output_Seq_Len, Output_Size)
        """
        # For this competition, we might just predict the next N frames based on the input sequence.
        # Or we might want a seq2seq encoder-decoder.
        # Given the simplicity, let's start with: 
        # Input: History -> LSTM -> Last Hidden State -> Decoder -> Future Trajectory
        # Or: Many-to-Many if we have targets for every input step?
        # The targets are FUTURE frames. The inputs are PAST frames.
        # So it's a Seq2Seq problem.
        
        # Simple approach:
        # Encode history.
        # Decode future.
        
        # However, standard LSTM takes input sequence and outputs sequence of same length.
        # If we want to predict M future frames from N past frames:
        # We can use an Encoder-Decoder architecture.
        
        # Let's assume for now we just want to output a fixed number of future steps.
        # But the number of output frames varies per play (num_frames_output).
        # We can just predict a fixed max length or use the specific length.
        
        # Let's stick to a simple Encoder-Decoder for now.
        pass
        
        # Actually, let's implement a simple Encoder-Decoder
        
        batch_size = x.size(0)
        
        # Encoder
        _, (hidden, cell) = self.lstm(x)
        
        # Decoder
        # We need to generate 'future_frames' steps.
        # Initial input to decoder could be the last observed position (x[:, -1, :2])
        # But we need full features for decoder input if we reuse the same LSTM?
        # Or we can have a separate Decoder LSTM.
        
        # Let's simplify: Just return the hidden state for now, 
        # and let the training loop handle the decoding or use a simpler Many-to-One -> FC -> Many approach?
        # No, trajectory implies sequence.
        
        # Let's use a simple approach where we predict 1 step ahead, 
        # and during inference we feed it back.
        # But during training we have the ground truth for the future?
        # The dataset gives us Input (Past) and Output (Future).
        
        # Let's implement a proper Encoder-Decoder.
        return self._forward_impl(x, future_frames)

    def _forward_impl(self, x, future_frames):
        # This is a placeholder.
        # For the initial implementation, let's just use the LSTM to encode 
        # and a linear layer to predict the *entire* future sequence at once?
        # No, that's fixed size.
        
        # Let's use the LSTM to output a sequence of length 'future_frames'.
        # We can repeat the context vector?
        
        # Better:
        # Encoder: Process input sequence.
        # Decoder: Unroll for future_frames.
        
        batch_size = x.size(0)
        
        # Encoder
        encoder_outputs, (hidden, cell) = self.lstm(x)
        
        # Decoder input: Start with zero or last position?
        # Let's just use the hidden state to predict the first future point, then feed it back?
        # Without teacher forcing for now to keep it simple.
        
        outputs = []
        
        # We need to know how many frames to predict. 
        # In training, we know from targets. In inference, we need to know.
        # Let's assume future_frames is passed.
        
        # Initial decoder input
        # We need to match input_size. 
        # If input has 15 features, decoder needs 15 features.
        # But we only predict x, y. We don't know future speed/roles etc?
        # Roles are constant. Speed can be estimated.
        # This suggests we need a separate Decoder that takes (x,y) + static features.
        
        # For V1, let's do a simple "Many-to-Many" where we just predict the next step
        # But wait, inputs and outputs are disjoint time segments.
        
        # Let's try a simple MLP decoder from the last hidden state.
        # Predicts 10 future frames (fixed) or something?
        # No, variable length is tricky with MLP.
        
        # Let's go with:
        # LSTM Encoder -> Hidden
        # LSTM Decoder -> Unroll
        
        # For the Decoder, we'll assume we feed in the LAST PREDICTED (x,y) 
        # plus the static features (roles) and maybe zero for dynamic ones?
        
        # To keep it extremely simple for the first iteration:
        # Just predict the NEXT frame.
        # But we need a sequence.
        
        # Let's just return the last hidden state for now and let a head predict the sequence?
        # Or just use a standard seq2seq.
        
        # Let's implement a basic loop.
        
        decoder_input = x[:, -1, :] # Last frame of input
        
        # We need to iterate
        preds = []
        
        curr_hidden = hidden
        curr_cell = cell
        curr_input = decoder_input
        
        for _ in range(future_frames):
            # Run LSTM cell (one step)
            # We need to use the LSTM layer but for one step.
            # nn.LSTM takes sequence. We can pass seq_len=1.
            
            out, (curr_hidden, curr_cell) = self.lstm(curr_input.unsqueeze(1), (curr_hidden, curr_cell))
            
            # Predict position
            pos_pred = self.fc(out.squeeze(1)) # (Batch, 2)
            preds.append(pos_pred)
            
            # Prepare next input
            # We need to construct the next input vector of size input_size.
            # We have pos_pred (x, y).
            # We need to fill the rest (s, a, dir, o, roles...).
            # For V1, let's just copy the previous input and update x, y?
            # This is a crude approximation but might work for a baseline.
            
            next_input = curr_input.clone()
            next_input[:, 0:2] = pos_pred # Update x, y
            # Ideally we update s, a, dir too based on change in x, y
            
            curr_input = next_input
            
        return torch.stack(preds, dim=1)
