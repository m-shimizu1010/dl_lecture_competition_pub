import torch

# モデルをロード
state_dict = torch.load("model.pth", map_location=torch.device('cpu'))

# 各レイヤーの名前とテンソルのshapeを表示
for name, param in state_dict.items():
    print(f"Layer: {name}, Shape: {param.shape}")