# EVAL.py
import sys, os, csv, torch, math, pandas as pd
from transformers import GPT2LMHeadModel
from tokenizers import Tokenizer

STIMULI_CSV = "data/waite_contexts_fixed.csv"
BLOCK_SIZE = 256
NUM_EXAMPLES = 5

# Word generation -> not really used in the end
def generate_next_word(model, tokenizer, context_ids, device, valid_first_tokens):
    ids = list(context_ids[-(BLOCK_SIZE - 10):]) # To leave room for generation
    generated = []

    for sub in range(10): # Maximum 10 subwords
        tensor = torch.tensor([ids], device = device)

        with torch.no_grad():
            logits = model(tensor).logits[0, -1, :] # Last token slot

        # To make sure model only guesses words starting with a letter (not a space)
        if sub == 0:
            mask = torch.full_like(logits, -float('inf'))
            mask[valid_first_tokens] = logits[valid_first_tokens]
            logits = mask

        next_id = torch.argmax(logits).item()
        token = tokenizer.id_to_token(next_id)
        if token is None:
            break

        if sub > 0 and "Ġ" in token: # Ġ is the start of a new word (byte level space prefix)
            break
        if sub > 0 and not any(c.isalpha() for c in token):
            break

        generated.append(next_id)
        ids.append(next_id)

    guess = tokenizer.decode(generated).strip()
    return guess

def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    tokenizer = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_first_tokens_list = []
    for i in range(tokenizer.get_vocab_size()):
        token = tokenizer.id_to_token(i)
        if token and any(char.isalpha() for char in token):
            valid_first_tokens_list.append(i)
    valid_first_tokens = torch.tensor(valid_first_tokens_list, device = device)

    items = []
    with open(STIMULI_CSV, "r", encoding = "utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            context = row.get("context")
            target = row.get("target_word")
            group = row.get("cloze_group")
            items.append((context.strip(), target.strip(), group))

    for stage in ("stage_child_best", "stage_adult_best"):
        stage_path = os.path.join(checkpoint_dir, stage)

        print(f"\n----- {stage} -----")

        model = GPT2LMHeadModel.from_pretrained(stage_path).to(device)
        model.eval()

        metrics = {
                "HC": {"count": 0, "gen_correct": 0, "top1_subw": 0, "top5_subw": 0, "top10_subw": 0, "surprisal": []},
                "LC": {"count": 0, "gen_correct": 0, "top1_subw": 0, "top5_subw": 0, "top10_subw": 0, "surprisal": []}}
        results = []
        csv_results = []

        for context, target, group in items:
            metrics[group]["count"] += 1
            context_ids = tokenizer.encode(context).ids
            target_tokens = tokenizer.encode(target.strip().lower()).ids
            first_token = target_tokens[0] if len(target_tokens) > 0 else None
            
            shortened_context = context_ids[-(BLOCK_SIZE - 10):] # To make room for target tokens

            with torch.no_grad():
                tensor_context = torch.tensor([shortened_context], device = device)
                logits = model(tensor_context).logits[0, -1, :]
                mask = torch.full_like(logits, -float('inf'))
                mask[valid_first_tokens] = logits[valid_first_tokens]
                logits = mask
                top5_ids = torch.topk(logits, 5).indices.tolist()
                top10_ids = torch.topk(logits, 10).indices.tolist()

                # Computing top-1, top-5, and top-10 accuracy on the first token
                if first_token:
                    if top5_ids[0] == first_token:
                        metrics[group]["top1_subw"] += 1
                    if first_token in top5_ids:
                        metrics[group]["top5_subw"] += 1
                    if first_token in top10_ids:
                        metrics[group]["top10_subw"] += 1

                # Computing surprisal
                word_prob = 1
                tmp_context = list(shortened_context)
                
                for t in target_tokens:
                    t_context_tensor = torch.tensor([tmp_context], device=device)
                    t_logits = model(t_context_tensor).logits[0, -1, :]
                    t_probs = torch.softmax(t_logits, dim=-1)
                    word_prob *= t_probs[t].item()
                    tmp_context.append(t)

                if word_prob > 0:
                    surprisal = -math.log2(word_prob)
                    metrics[group]["surprisal"].append(surprisal)
                
            # Generation of target word
            pred_word = generate_next_word(model, tokenizer, context_ids, device, valid_first_tokens)
            is_correct = (pred_word.lower() == target.strip().lower())
            if is_correct:
                metrics[group]["gen_correct"] += 1

            results.append((context, target, group, pred_word, is_correct))
            csv_results.append({"target": target, "pred": pred_word.strip().lower(), "group": group, "surprisal": surprisal, 
                "top1": (first_token == top5_ids[0]), "top5": (first_token in top5_ids), "top10": (first_token in top10_ids)})

        output_file = os.path.join(checkpoint_dir, f"results_{stage}.csv")
        pd.DataFrame(csv_results).to_csv(output_file, index=False)

        # Printing everything
        total = metrics["HC"]["count"] + metrics["LC"]["count"]
        total_correct = metrics["HC"]["gen_correct"] + metrics["LC"]["gen_correct"]
        print(f"Generated words accuracy: {total_correct}/{total} = {total_correct/total:.3f}")

        for group in ["HC", "LC"]:
            count = metrics[group]["count"]
            if count == 0: 
                continue

            gen_acc = metrics[group]["gen_correct"] / count
            top1_acc = metrics[group]["top1_subw"] / count
            top5_acc = metrics[group]["top5_subw"] / count
            top10_acc = metrics[group]["top10_subw"] / count
            avg_surp = sum(metrics[group]["surprisal"]) / len(metrics[group]["surprisal"]) if metrics[group]["surprisal"] else 0

            print(f"\n  {group} cloze category")
            print(f"    - Word generation (exact match): {metrics[group]['gen_correct']}/{count} = {gen_acc:.3f}")
            print(f"    - Accuracy (first token): top-1: {metrics[group]['top1_subw']}/{count} = {top1_acc:.3f}, top-5: {metrics[group]['top5_subw']}/{count} = {top5_acc:.3f}, top-10: {metrics[group]['top10_subw']}/{count} = {top10_acc:.3f}")
            print(f"    - Average surprisal (in bits): {avg_surp:.2f}")

        print(f"\nFirst {NUM_EXAMPLES} examples")
        printed = 0
        for context, target, group, pred_word, is_correct in results:
            if printed >= NUM_EXAMPLES:
                break
            short_context = context[-40:] if len(context) > 40 else context
            print(f"\nLast context: ...{short_context}")
            print(f"Target: '{target}' --- Model generated: '{pred_word}' --- {'TRUE' if is_correct else 'FALSE'}")
            printed += 1
        print("\n")

if __name__ == "__main__":
    main()
