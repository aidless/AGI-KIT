from __future__ import annotations


def test_sft_labels_mask_padding_tokens():
    input_ids = [[12, 2, 2], [7, 8, 2]]
    attention_mask = [[1, 0, 0], [1, 1, 0]]
    labels = [
        [token if mask else -100 for token, mask in zip(ids, attention)]
        for ids, attention in zip(input_ids, attention_mask)
    ]
    assert labels == [[12, -100, -100], [7, 8, -100]]
