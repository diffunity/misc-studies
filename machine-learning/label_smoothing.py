import torch

def label_smoothing_loss(logits, target, input_ids, attention_mask, label_smoothing):
    # code revised from : https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186
    assert 0.0 < label_smoothing <= 1.0

    smoothing_value = label_smoothing / (input_ids.shape[1] - 1)

    one_hot = torch.full((input_ids.shape[1],), smoothing_value)

    confidence = 1.0 - label_smoothing

    model_prob = one_hot.repeat(target.size(0), 1).to(device)
    model_prob.scatter_(1, target, confidence)
    model_prob[attention_mask == 0] = 0.0

    return F.kl_div(torch.log_softmax(logits, 1), model_prob, reduction="batchmean")
    # return F.kl_div(torch.log_softmax(logits, 1), model_prob, reduction = "sum")