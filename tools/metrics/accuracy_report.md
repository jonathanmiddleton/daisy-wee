### Accuracy Report Summary
The script builds the reference continuation tokens (`ref_ids`) in this order:
- If you pass `--reference_text`, it tokenizes that text (GPT‑2 BPE via `tiktoken`) and truncates to `--max_new_tokens`.
- Else if you pass `--reference_file`, it searches that file for the first occurrence of your exact `--prompt` text and uses the text immediately after that occurrence as the reference continuation (tokenized and truncated to `--max_new_tokens`).
- Else it tries the bundled `data/the_time_machine.txt` in the same way (find prompt → take the following text); if nothing is found, no reference is used and accuracy metrics are skipped.

Notes:
- The prompt itself is tokenized and (if needed) truncated to the model’s attention window before generation, but the reference continuation is limited by `--max_new_tokens`.
- Tokenization is consistent throughout using GPT‑2 encoding.

### What gets compared to the reference
- The model generates up to `--max_new_tokens` tokens from the prompt on each device (CPU and MPS). Those generated tokens are compared against `ref_ids`.

### Metrics computed against the reference
If a reference exists, the following are computed per device:
- `token_accuracy`: exact token‑by‑token match rate over the overlapping length.
- `bleu1`: unigram BLEU (with brevity penalty) between generated tokens and reference tokens.
- `rougeL_f1`: ROUGE‑L F1 based on the longest common subsequence.
- `avg_nll` and `perplexity`: negative log‑likelihood and PPL of the reference continuation under the model conditioned on the prompt (teacher forcing). This is done by pre‑filling on the prompt, then stepping through each reference token and summing `-log_softmax(logits)[y]`.

### What is not a reference
- The CPU vs MPS "consistency" section is a separate side‑by‑side check (exact match and token agreement between the two generations). It is not used as the ground truth for accuracy.

### Bottom line
- Reference points = the ground‑truth continuation tokens after your prompt, sourced from `--reference_text`, or from `--reference_file` (text after the prompt in that file), or from `data/the_time_machine.txt` if available. All accuracy metrics compare the model’s generated continuation to those reference tokens.