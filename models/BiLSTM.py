import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(BiLSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # embedded = self.dropout(self.embedding(text))
        embedded = self.dropout(text.float())
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Concatenate the final forward and backward hidden states
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.fc(hidden)
        return output

def get_bilstm_model(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional=True, dropout=0.5):
    """
    创建并返回一个 BiLSTM 模型。

    参数:
    - vocab_size (int): 词汇表大小。
    - embedding_dim (int): 嵌入层维度。
    - hidden_dim (int): LSTM 隐藏层维度。
    - output_dim (int): 输出层维度，通常为分类任务的类别数。
    - n_layers (int): LSTM的层数。
    - bidirectional (bool): 是否使用双向LSTM。
    - dropout (float): Dropout率，用于防止过拟合。

    返回:
    - model (BiLSTMModel): 配置好的 BiLSTM 模型实例。
    """
    model = BiLSTMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        bidirectional=bidirectional,
        dropout=dropout
    )
    return model