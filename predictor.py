import torch
from dataset import MyDataset


def predict(model, video, coords, length):
    with torch.inference_mode():
        output = model(
            video.unsqueeze(0),
            coords.unsqueeze(0),
        )

    token_ids = output[0, :length].argmax(dim=-1).tolist()
    return MyDataset.ctc_arr2txt(token_ids, start=1)