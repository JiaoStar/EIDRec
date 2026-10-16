# EIDRec

EIDRec is a sequential recommendation model implementation based on PyTorch. The code trains and evaluates the model on user interaction sequences and reports standard top-k recommendation metrics such as HR and NDCG.

## Directory Structure

```text
EIDRec/
├── EIDRec_main.py          # Training and evaluation entry point
├── EIDRec_models.py        # EIDRec model definition
├── EIDRec_trainer.py       # Training, validation, and test loops
├── datasets.py             # Dataset preprocessing and PyTorch Dataset
├── modules.py              # Transformer/SASRec building blocks
├── utils.py                # Utilities, logging, metrics, early stopping
├── environment.yml         # Conda environment file
└── scripts/
    ├── EIDRec_run_all.sh   # Run EIDRec on all supported datasets
    └── run_sampling_fix.sh # Single-dataset or all-dataset runner with env defaults
```

## Environment

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate eidrec
```

## Data Preparation

Place datasets under `./datasets/`:

```text
datasets/
├── Beauty.txt
├── Toys_and_Games.txt
├── Sports_and_Outdoors.txt
└── Yelp.txt
```

Each line should contain one user and the user's chronological item sequence:

```text
user_id item_id item_id item_id ...
```

Example:

```text
0 12 53 89 104 231
1 7 19 22 95
```


## Quick Start

Train and evaluate EIDRec on Beauty:

```bash
python EIDRec_main.py \
  --data_name Beauty \
  --model_name EIDRec \
  --output_dir output \
  --cuda 0
```

Run a specific dataset:

```bash
DATA_NAME=Yelp bash scripts/run_sampling_fix.sh
```

Run all supported datasets:

```bash
RUN_ALL=1 bash scripts/run_sampling_fix.sh
```

Run the full benchmark script:

```bash
bash scripts/EIDRec_run_all.sh
```

## Evaluation Only

After training, evaluate from a saved checkpoint:

```bash
python EIDRec_main.py \
  --data_name Beauty \
  --model_name EIDRec \
  --model_idx 0 \
  --output_dir output \
  --eval_only
```

The checkpoint path is inferred as:

```text
output/EIDRec-Beauty-0.pt
```

## Resume Training

Training periodically writes a resume checkpoint:

```text
output/EIDRec-Beauty-0.resume.pt
```

Resume training with:

```bash
python EIDRec_main.py \
  --data_name Beauty \
  --model_name EIDRec \
  --model_idx 0 \
  --output_dir output \
  --resume
```

You can also specify a custom resume checkpoint:

```bash
python EIDRec_main.py \
  --data_name Beauty \
  --resume \
  --resume_checkpoint_path output/EIDRec-Beauty-0.resume.pt
```

## Important Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--data_dir` | `./datasets/` | Directory containing dataset text files |
| `--data_name` | `Beauty` | Dataset name without `.txt` |
| `--output_dir` | `output` | Directory for logs and checkpoints |
| `--model_name` | `EIDRec` | Model name used in output filenames |
| `--model_idx` | `0` | Run identifier |
| `--cuda` | `0` | CUDA device id |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `256` | Batch size |
| `--max_seq_length` | `50` | Maximum sequence length |
| `--hidden_size` | `64` | Item embedding and hidden dimension |
| `--intent_num` | `256` | Number of intent clusters |
| `--diff_weight` | `1` | Weight of diffusion loss |
| `--cl_weight` | `0.2` | Weight of contrastive loss |
| `--timesteps` | `200` | Diffusion timesteps |
| `--sample_steps` | `50` | Reverse diffusion sampling steps |
| `--w` | `2.0` | Intent guidance strength |
| `--eval_only` | `False` | Only run testing from checkpoint |
| `--resume` | `False` | Resume from full-training checkpoint |

## Outputs

For a run with `--model_name EIDRec --data_name Beauty --model_idx 0`, outputs are saved as:

```text
output/EIDRec-Beauty-0.txt        # Training log and final metrics
output/EIDRec-Beauty-0.pt         # Best model checkpoint
output/EIDRec-Beauty-0.resume.pt  # Resume checkpoint
```

`scripts/EIDRec_run_all.sh` also writes:

```text
output/EIDRec_results.csv
```

## Reproducibility

The default random seed is `2022`. To use a different seed:

```bash
python EIDRec_main.py --data_name Beauty --seed 2023
```

## Notes

- Use `--no-eval_topk_on_gpu` if GPU memory is limited during evaluation.
- Use `--no-spectral_view_filter` to disable spectral filtering for generated views.
- Use `--without_segment` to generate randomly shuffled segmented training sequences.
