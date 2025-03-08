import argparse


def parser():
    parser = argparse.ArgumentParser("multimodal medical pretraining")

    # Experiment
    parser.add_argument("--exp_name", type=str, default="multimodal-mimic-3-pretraining-epoch-200")
    parser.add_argument(
        "--mimic_root", type=str, default="/workspaces/multimodal-mimic/raw-mimic3"
    )
    parser.add_argument(
        "--mimic_benchmark_root",
        type=str,
        default="/workspaces/multimodal-mimic/mimic3-benchmarks",
    )

    # Distributed
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--dist-url", default="env://")

    # Evaluation
    parser.add_argument("--k_test", default=256)
    parser.add_argument("--evaluation", action="store_true")

    # Optimizer and schedular
    parser.add_argument("--opt", default="adamW")
    parser.add_argument("--sched", default="cosine")
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--min_lr", default=1e-8, type=float)
    parser.add_argument("--warmup", default=True, type=bool)
    parser.add_argument("--warmup_lr", default=0.00001, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--decay_rate", default=1, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--warmup_epochs", default=0, type=int)
    parser.add_argument("--cooldown_epochs", default=0, type=int)
    parser.add_argument("--report_freq", type=float, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    # Loss
    parser.add_argument(
        "--loss",
        choices=["CLIP", "SogCLR", "iSogCLR", "VICReg", "MSE", "CLIP+MSE"], default="CLIP+MSE",
    )
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--learnable_temp", action="store_true")

    # Multimodal Model
    parser.add_argument("--use_cls_token", action="store_true", default=True)
    parser.add_argument(
        "--mlp",
        default="128",
    )

    # Measurement Model
    parser.add_argument("--use_measurements", action="store_true", default=True)
    parser.add_argument("--measurement_emb_size", type=int, default=128)
    parser.add_argument("--measurement_num_heads", type=int, default=8)
    parser.add_argument("--measurement_num_layers", type=int, default=8)
    parser.add_argument("--use_pos_emb", action="store_true", default=True)
    parser.add_argument("--measurement_dropout", type=float, default=0.1)
    parser.add_argument("--measurement_max_seq_len", type=int, default=256)
    parser.add_argument("--measurement_mask_rate", type=float, default=0.1)
    parser.add_argument(
        "--measurement_model",
        default="Transformer",
        choices=["Transformer"],
    )
    parser.add_argument(
        "--measurement_activation", choices=["ReLU", "GELU"], default="GELU"
    )

    # Notes Model
    parser.add_argument("--use_notes", action="store_true", default=True)
    parser.add_argument("--notes_emb_size", type=int, default=128)
    parser.add_argument("--notes_num_heads", type=int, default=8)
    parser.add_argument("--notes_num_layers", type=int, default=8)
    parser.add_argument("--notes_dropout", type=float, default=0.1)
    parser.add_argument("--notes_max_seq_len", type=int, default=256)
    parser.add_argument("--notes_mask_rate", type=float, default=0.1)
    parser.add_argument(
        "--text_model",
        default="BERT",
        choices=[
            "BERT",
            "ClinicalBERT",
        ],
    )

    parser.add_argument("--device", default="cuda")

    return parser.parse_args()