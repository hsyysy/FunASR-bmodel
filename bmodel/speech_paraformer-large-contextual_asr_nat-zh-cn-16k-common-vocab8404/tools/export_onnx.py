import torch
import torch.onnx
import torch.nn as nn

def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad

inner_dim = 512
bias_encoder_dropout_rate = 0.0
vocab_size = 8404

model_path = "speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, inner_dim):
        super(EmbeddingModel, self).__init__()
        self.bias_embed = torch.nn.Embedding(vocab_size, inner_dim)
        self.bias_embed.weight = torch.load(model_path + "/embedding_weight.pt",weights_only=True)

    def forward(self, x):
        return self.bias_embed(x)

model = EmbeddingModel(vocab_size=vocab_size, inner_dim=inner_dim)

hw_list = [ torch.Tensor([1]).long() ]
input_tensor = pad_list(hw_list, 0)

model.eval()

torch.onnx.export(model, input_tensor, "embedding.onnx", opset_version=11)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.bias_encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0)
        self.bias_encoder._parameters['weight_ih_l0'] = torch.load(model_path+"/weight_ih_l0.pt",weights_only=True)
        self.bias_encoder._parameters['weight_hh_l0'] = torch.load(model_path+"/weight_hh_l0.pt",weights_only=True)
        self.bias_encoder._parameters['bias_ih_l0']   = torch.load(model_path+"/bias_ih_l0.pt",weights_only=True)
        self.bias_encoder._parameters['bias_hh_l0']   = torch.load(model_path+"/bias_hh_l0.pt",weights_only=True)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.bias_encoder(x)
        return h_n

model = LSTMModel(inner_dim, inner_dim, 1)

input_tensor = torch.randn(1, 1, inner_dim)

# 设置模型为评估模式
model.eval()

# 导出模型到 ONNX 格式
torch.onnx.export(model, input_tensor, "lstm.onnx", opset_version=11)
