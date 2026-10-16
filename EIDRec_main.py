import os
import numpy as np
import torch
import argparse
import random

import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datetime import datetime

from EIDRec_models import EIDRec
from utils import EarlyStopping, set_seed, check_path, show_args_info
from datasets import D,D_random, get_seqs_and_matrixes, DatasetForEIDRec
from EIDRec_trainer import EIDRecTrainer



def get_args():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--model_name", default="EIDRec", type=str)
    parser.add_argument("--data_dir", default="./datasets/", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="resume training from the latest full-training checkpoint")
    parser.add_argument("--resume_checkpoint_path", default=None, type=str,
                        help="path to the full-training checkpoint used by --resume")
    parser.add_argument("--save_resume_every", default=1, type=int,
                        help="save a full-training checkpoint every N epochs")
    parser.add_argument("--model_idx", default=0, type=int, 
                        help="model idenfier 10, 20, 30...")
    parser.add_argument('--cuda', type=int, default=0,
                            help='cuda device.')

    # robustness experiments
    parser.add_argument("--noise_ratio", type=float, default=0.0,
                        help="percentage of negative interactions in a sequence",)

    ## contrastive learning task args
    parser.add_argument("--temperature", default=1.0, type=float,
                        help="softmax temperature (default:  1.0).")
    parser.add_argument("--intent_num",default=256,type=int,
                        help="the multi intent nums.")
    parser.add_argument("--sim",default='dot',type=str,
                        help="the calculate ways of the similarity.")

    # model args
    parser.add_argument("--hidden_size", type=int, default=64, 
                        help="Number of hidden factors, i.e., embedding size.")
    parser.add_argument("--num_hidden_layers", type=int, default=2, 
                        help="number of layers")
    parser.add_argument("--num_attention_heads", type=int, default=2,
                        help="number of attention heads")
    parser.add_argument("--hidden_act", type=str, default="gelu", 
                        help="active function type.") 

    # dropout
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, 
                        help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, 
                        help="hidden dropout p")
    parser.add_argument('--p', type=float, default=0.1,
                        help='uncondition dropout')

    

    # train args
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="number of epochs")
    parser.add_argument("--print_log_freq", type=int, default=1, 
                        help="per epoch print res")
    parser.add_argument("--cluster_update_interval", type=int, default=3,
                        help="rebuild KMeans intent clusters every N epochs")
    parser.add_argument("--eval_topk_on_gpu", action=argparse.BooleanOptionalAction, default=True,
                        help="keep full-sort masking and top-k selection on GPU during evaluation")
    parser.add_argument("--max_seq_length", type=int, default=50, 
                        help="max sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="weight_decay of adam")
    parser.add_argument("--seed", default=2022, type=int)

    parser.add_argument("--initializer_range", type=float, default=0.02)


    # loss weight
    parser.add_argument("--rec_weight", type=float, default=1, 
                        help="weight of rating prediction")
    parser.add_argument("--diff_weight", type=float, default=1, 
                        help="weight of intent-aware diffusion task")
    parser.add_argument("--cl_weight", type=float, default=0.2, 
                        help="weight of contrastive learning task")
    parser.add_argument("--spectral_view_filter", action=argparse.BooleanOptionalAction, default=True,
                        help="enable spectral diffusion multiplier filtering for generated views")
    parser.add_argument("--spectral_tau_init", type=float, default=0.1,
                        help="initial value of learnable diffusion time tau for each hidden channel")
    parser.add_argument("--spectral_lambda_scale", type=float, default=1.0,
                        help="global scale for frequency eigenvalues in spectral filter")
    parser.add_argument("--spectral_filter_residual", type=float, default=0.5,
                        help="residual mixing ratio between filtered and original generated view")

    # ablation study
    parser.add_argument('--without_segment', action="store_true",
                        help='dropout ')

    # control guidance signal
    parser.add_argument('--w', type=float, default=2.0,
                        help='control the strength of intent-guided signal s')
    parser.add_argument("--guide_entropy_lambda", type=float, default=0.1,
                        help="entropy penalty strength for entropy-aware guide pooling")
    parser.add_argument("--guide_entropy_temperature", type=float, default=1.0,
                        help="temperature for entropy-aware guide pooling")
    parser.add_argument("--guide_mode", choices=["entropy", "last"], default="entropy",
                        help="intent guidance representation used by diffusion")

    # diffusion
    parser.add_argument('--timesteps', type=int, default=200,
                        help='timesteps for diffusion')
    parser.add_argument('--sample_steps', type=int, default=50,
                        help='reverse diffusion steps used to generate training CL views; <=0 uses all timesteps')
    parser.add_argument('--sample_eta', type=float, default=1.0,
                        help='stochasticity for skipped reverse diffusion; 0.0 is deterministic DDIM')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='beta start of diffusion')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='type of diffuser.')
    parser.add_argument('--beta_sche', nargs='?', default='exp',
                        help='beta schedule')
    

    return parser.parse_args()


def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if torch.is_tensor(torch_state):
        torch_state = torch_state.detach().cpu().to(torch.uint8)
    else:
        torch_state = torch.as_tensor(torch_state, dtype=torch.uint8).cpu()
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = []
        for cuda_state in state["cuda"]:
            if torch.is_tensor(cuda_state):
                cuda_state = cuda_state.detach().cpu().to(torch.uint8)
            else:
                cuda_state = torch.as_tensor(cuda_state, dtype=torch.uint8).cpu()
            cuda_states.append(cuda_state)
        torch.cuda.set_rng_state_all(cuda_states)


def early_stopping_state_dict(early_stopping):
    return {
        "counter": early_stopping.counter,
        "best_score": early_stopping.best_score,
        "early_stop": early_stopping.early_stop,
        "score_min": getattr(early_stopping, "score_min", None),
    }


def load_early_stopping_state(early_stopping, state):
    if not state:
        return
    early_stopping.counter = state.get("counter", 0)
    early_stopping.best_score = state.get("best_score")
    early_stopping.early_stop = state.get("early_stop", False)
    if "score_min" in state:
        early_stopping.score_min = state["score_min"]


def save_resume_checkpoint(path, epoch, trainer, early_stopping, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "early_stopping": early_stopping_state_dict(early_stopping),
            "rng_state": get_rng_state(),
            "args": vars(args),
        },
        path,
    )


def load_resume_checkpoint(path, trainer, early_stopping, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    load_early_stopping_state(early_stopping, checkpoint.get("early_stopping"))
    set_rng_state(checkpoint.get("rng_state"))
    return int(checkpoint["epoch"]) + 1


if __name__ == '__main__':
    args = get_args()

    set_seed(args.seed)
    check_path(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    print("Using Cuda:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.data_file = args.data_dir + args.data_name + ".txt"

    if args.without_segment:
        args.segmented_file = args.data_dir + args.data_name + "_random.txt"
        D_random(args.data_file,args.segmented_file,args.max_seq_length)

    else:
        args.segmented_file = args.data_dir + args.data_name + "_s.txt"
    
    if not os.path.exists(args.segmented_file):
        D(args.data_file,args.segmented_file,args.max_seq_length)

    _,train_seq = get_seqs_and_matrixes("training", args.segmented_file)
    _,user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_seqs_and_matrixes("rating", args.data_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    args_str = f"{args.model_name}-{args.data_name}-{args.model_idx}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    if args.resume_checkpoint_path is None:
        args.resume_checkpoint_path = os.path.join(args.output_dir, args_str + ".resume.pt")

    show_args_info(args)
    with open(args.log_file, "a") as f: 
        f.write(str(args) + "\n")

    if args.eval_only:
        cluster_dataloader, training_dataloader, eval_dataloader = [],[],[]
        testing_dataset = DatasetForEIDRec(args, user_seq, data_type="test")
        testing_sampler = RandomSampler(testing_dataset)
        testing_dataloader = DataLoader(testing_dataset, sampler=testing_sampler, batch_size=args.batch_size, drop_last=True)
    else:
        args.train_matrix = valid_rating_matrix

        cluster_dataset = DatasetForEIDRec(args, train_seq, data_type="train")
        cluster_sampler = SequentialSampler(cluster_dataset)
        cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size, drop_last=True)

        training_dataset = DatasetForEIDRec(args, train_seq, data_type="train")
        training_sampler = RandomSampler(training_dataset)
        training_dataloader = DataLoader(training_dataset, sampler=training_sampler, batch_size=args.batch_size, drop_last=True)

        eval_dataset = DatasetForEIDRec(args, user_seq, data_type="valid")
        eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, drop_last=True)

        testing_dataset = DatasetForEIDRec(args, user_seq, data_type="test")
        testing_sampler = RandomSampler(testing_dataset)
        testing_dataloader = DataLoader(testing_dataset, sampler=testing_sampler, batch_size=args.batch_size, drop_last=True)


    model = EIDRec(device, args=args)

    trainer = EIDRecTrainer(model, training_dataloader, cluster_dataloader, eval_dataloader, testing_dataloader, device, args)

    model.to(device)
    

    if args.eval_only:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)

        print(f"==================== Loading trained model from {args.checkpoint_path} for test...")
        scores, log_info = trainer.test(0) # set epoch equal 0 to print results directly.

    else:
        print(f"==================== Training ...")
        start_time = datetime.now()
        with open(args.log_file, "a") as f:
            f.write(f"====================  Training EIDRec at: {start_time} ====================" + "\n")
        
        early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        start_epoch = 0
        if args.resume:
            if not os.path.exists(args.resume_checkpoint_path):
                raise FileNotFoundError(f"Missing resume checkpoint: {args.resume_checkpoint_path}")
            start_epoch = load_resume_checkpoint(args.resume_checkpoint_path, trainer, early_stopping, device)
            print(f"==================== Resumed from {args.resume_checkpoint_path}, start_epoch={start_epoch}")
            with open(args.log_file, "a") as f:
                f.write(
                    f"====================  Resumed from {args.resume_checkpoint_path}, "
                    f"start_epoch:{start_epoch} at: {datetime.now()} ====================\n"
                )

        for epoch in range(start_epoch, args.epochs):
            trainer.train(epoch)

            if epoch % 5 == 0:
                scores, _ = trainer.valid(epoch)
                # HR@20, ND@20
                early_stopping(np.array(scores[-2:]), trainer.model)
                if early_stopping.early_stop:
                    save_resume_checkpoint(args.resume_checkpoint_path, epoch, trainer, early_stopping, args)
                    print("==================== Early stopping triggered!")
                    break

            if args.save_resume_every > 0 and (epoch + 1) % args.save_resume_every == 0:
                save_resume_checkpoint(args.resume_checkpoint_path, epoch, trainer, early_stopping, args)

        trainer.args.train_matrix = test_rating_matrix
        print("====================  Testing Phase  ====================")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)
        
        end_time = datetime.now()
        cost_time = str(end_time-start_time)
        print(args_str)
        print(result_info)
        with open(args.log_file, "a") as f:
            f.write(f"Finished at: {end_time} ==============================================================================================================" + "\n")
            f.write(f" diff_weight:{args.diff_weight} || cl_weight:{args.cl_weight} || spectral_view_filter:{args.spectral_view_filter} || spectral_tau_init:{args.spectral_tau_init} || spectral_lambda_scale:{args.spectral_lambda_scale} || spectral_filter_residual:{args.spectral_filter_residual} || guide_mode:{args.guide_mode} || guide_entropy_lambda:{args.guide_entropy_lambda} || guide_entropy_temperature:{args.guide_entropy_temperature} || intent_num: {args.intent_num} || dropout: {args.hidden_dropout_prob} || w: {args.w} || p: {args.p} || timesteps: {args.timesteps} || sample_steps: {args.sample_steps} || sample_eta: {args.sample_eta} || cluster_update_interval: {args.cluster_update_interval} || eval_topk_on_gpu: {args.eval_topk_on_gpu} || time_cost: {cost_time}" + "\n")
            f.write(result_info + "\n")
            f.write("======================================================================================================================================================" + "\n")
