import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from main import VQADataset, VQAModel
from tqdm import tqdm

def load_model_and_predict(model_path, test_data_path, image_dir, output_path):
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # テストデータセットの準備
    test_dataset = VQADataset(df_path=test_data_path, image_dir=image_dir, transform=transform, answer=False)
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset.update_dict(train_dataset)
    
    # データローダーの準備
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # モデルの準備
    vocab_size = len(test_dataset.question2idx) + 1  # +1 for unknown tokens
    # model = VQAModel(vocab_size=vocab_size, n_answer=len(test_dataset.answer2idx)).to(device)
    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx), 
                 d_model=512, nhead=8, num_encoder_layers=6).to(device)
    # 保存されたモデルの重みを読み込む
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 推論の実行
    predictions = []
    with torch.no_grad():
        for image, question in tqdm(test_loader):
            image, question = image.to(device), question.to(device)
            output = model(image, question)
            print((output[0])[:5])
            pred = output.argmax(1).cpu().item()
            print("pred",pred)
            print(train_dataset.idx2answer[pred])
            predictions.append(pred)
        
    # 予測結果をラベルに変換
    predictions = [train_dataset.idx2answer[id] for id in predictions]
    predictions = np.array(predictions)

    # 結果を保存
    np.save(output_path, predictions)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    model_path = "model.pth"  # 保存したモデルのパス
    test_data_path = "./data/valid.json"  # テストデータのパス
    image_dir = "./data/valid"  # テスト画像のディレクトリ
    output_path = "submission.npy"  # 出力ファイルのパス

    load_model_and_predict(model_path, test_data_path, image_dir, output_path)