import argparse
import json
import math
import re
from collections import Counter


def sid_tokens(sid):
    return re.findall(r"<[abc]_[^>]+>", str(sid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    with open(args.predictions) as f:
        data = json.load(f)

    with open(args.index) as f:
        item_to_tokens = json.load(f)

    valid_sids = {
        "".join(tokens)
        for tokens in item_to_tokens.values()
    }
    catalog_a = {
        tokens[0]
        for tokens in item_to_tokens.values()
        if tokens
    }

    num_users = len(data)
    topk = args.topk

    a_hit_1 = 0
    a_hit_k = 0
    ab_hit_1 = 0
    ab_hit_k = 0
    sid_hit_1 = 0
    sid_hit_k = 0
    ndcg_k = 0.0

    valid_count = 0
    total_predictions = 0
    unique_valid_sids = set()
    predicted_a = Counter()

    for sample in data:
        target = "".join(sid_tokens(sample["output"]))
        target_tokens = sid_tokens(target)

        predictions = [
            "".join(sid_tokens(pred))
            for pred in sample["predict"][:topk]
        ]
        prediction_tokens = [sid_tokens(pred) for pred in predictions]

        if predictions:
            top1 = predictions[0]
            top1_tokens = prediction_tokens[0]

            a_hit_1 += int(
                len(target_tokens) >= 1
                and len(top1_tokens) >= 1
                and target_tokens[:1] == top1_tokens[:1]
            )
            ab_hit_1 += int(
                len(target_tokens) >= 2
                and len(top1_tokens) >= 2
                and target_tokens[:2] == top1_tokens[:2]
            )
            sid_hit_1 += int(top1 == target)

        a_hit_k += int(any(
            len(tokens) >= 1 and tokens[:1] == target_tokens[:1]
            for tokens in prediction_tokens
        ))
        ab_hit_k += int(any(
            len(tokens) >= 2 and tokens[:2] == target_tokens[:2]
            for tokens in prediction_tokens
        ))
        sid_hit_k += int(target in predictions)

        if target in predictions:
            rank = predictions.index(target)
            ndcg_k += 1.0 / math.log2(rank + 2)

        for pred, tokens in zip(predictions, prediction_tokens):
            total_predictions += 1

            if tokens:
                predicted_a[tokens[0]] += 1

            if pred in valid_sids:
                valid_count += 1
                unique_valid_sids.add(pred)

    if total_predictions:
        entropy = -sum(
            (count / total_predictions)
            * math.log(count / total_predictions)
            for count in predicted_a.values()
        )
    else:
        entropy = 0.0

    normalized_entropy = (
        entropy / math.log(len(catalog_a))
        if len(catalog_a) > 1
        else 0.0
    )

    dominant_a, dominant_count = (
        predicted_a.most_common(1)[0]
        if predicted_a
        else ("NONE", 0)
    )

    print("========== Unified Evaluation ==========")
    print("Users:", num_users)
    print("Top-K:", topk)
    print("A-Hit@1:", a_hit_1 / num_users)
    print(f"A-Hit@{topk}:", a_hit_k / num_users)
    print("AB-Hit@1:", ab_hit_1 / num_users)
    print(f"AB-Hit@{topk}:", ab_hit_k / num_users)
    print("SID-Hit@1:", sid_hit_1 / num_users)
    print(f"SID-Hit@{topk}:", sid_hit_k / num_users)
    print(f"NDCG@{topk}:", ndcg_k / num_users)
    print(
        f"Valid SID Rate@{topk}:",
        valid_count / total_predictions if total_predictions else 0.0
    )
    print(
        f"Coverage@{topk}:",
        len(unique_valid_sids) / len(valid_sids)
    )
    print(
        "A-layer Coverage:",
        len(predicted_a) / len(catalog_a)
    )
    print("A-layer Normalized Entropy:", normalized_entropy)
    print(
        "Dominant A Token:",
        dominant_a,
        "share=",
        dominant_count / total_predictions
        if total_predictions else 0.0
    )
    print("Unique recommended items:", len(unique_valid_sids))
    print("Catalog items:", len(valid_sids))

    print("\nMost common A tokens:")
    for token, count in predicted_a.most_common(10):
        print(
            token,
            "count=",
            count,
            "share=",
            round(count / total_predictions, 4)
        )


if __name__ == "__main__":
    main()
