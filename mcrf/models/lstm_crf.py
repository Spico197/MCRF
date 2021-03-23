import torch
import torch.nn as nn


class LSTMCRFModel(nn.Module):
    def __init__(self, config, crf) -> None:
        super().__init__()
        self.config = config

        self.emb_enc = nn.Embedding(self.config.vocab_size, self.config.emb_size)
        self.lstm_enc = nn.LSTM(
            self.config.emb_size,
            self.config.hidden_size,
            num_layers=self.config.num_lstm_layers,
            bias=True,
            batch_first=True,
            dropout=self.config.dropout,
            bidirectional=True
        )
        self.hidden2tag = nn.Linear(2 * self.config.hidden_size, self.config.num_tags)
        self.dropout = nn.Dropout(self.config.dropout)
        self.crf = crf

    def forward(self, inputs, tags=None, mask=None, **kwargs):
        emb = self.emb_enc(inputs)
        emb = self.dropout(emb)
        sorted_seq_lengths, indices = torch.sort(mask.sum(-1), descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        emb = emb[indices]
        out = nn.utils.rnn.pack_padded_sequence(emb, sorted_seq_lengths.detach().cpu(), batch_first=True)
        out, (_, _) = self.lstm_enc(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=mask.size(-1))
        out = out[desorted_indices]
        out = self.hidden2tag(out)

        if mask is not None:
            mask = mask.bool()
        if tags is not None:
            results = -self.crf(out, tags, mask)
        else:
            results = self.crf.decode(out)

        return results
