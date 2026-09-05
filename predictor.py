import torch
from dataset import MyDataset


def predict(model, video, coords, length):
    with torch.inference_mode():
        output = model(
            video.unsqueeze(0),
            coords.unsqueeze(0),
        )

    logits = output[0, :length]
    probabilities = torch.softmax(logits, dim=-1)

    token_probabilities, token_ids = probabilities.max(dim=-1)
    non_blank = token_ids >= 1

    if non_blank.any():
        confidence = token_probabilities[non_blank].mean().item()
    else:
        confidence = 0.0

    prediction = MyDataset.ctc_arr2txt(token_ids.tolist(), start=1)
    confidence_percent = round(confidence * 100, 2)

    return prediction, confidence_percent