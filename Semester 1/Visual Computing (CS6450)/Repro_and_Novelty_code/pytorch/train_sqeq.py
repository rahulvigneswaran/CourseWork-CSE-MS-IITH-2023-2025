import argparse
from genericpath import exists
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from experiment_utils.generate_data import data_loader
from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel

datasets = ['sqeq', 'sqeq_d', 'sqeq_cd', 'sqeq_cd_eos']

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='reverse',
                    choices=datasets,
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=100000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--num_mem_tokens', type=int, default=0,
                    help='number of memory tokens in work memory')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval')
parser.add_argument('--work_dir', default='LM-TFM', type=str,
                    help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--sample_softmax', type=int, default=-1,
                    help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--finetune_v2', action='store_true',
                    help='finetune v2')
parser.add_argument('--finetune_v3', action='store_true',
                    help='finetune v3')
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can '
                    'improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument'
                    ' supersedes --static-loss-scale.')
parser.add_argument('--device_ids', nargs='+', default=None, help='Device ids for training.')
parser.add_argument('--mem_backprop_depth', type=int, default=0, 
                    help='How deep to pass gradient with memory tokens to past segments .')
parser.add_argument('--bptt_bp', action='store_true',
                    help='Backpropagate at each timestep during BPTT.')
parser.add_argument('--mem_at_end', action='store_true',
                    help='Whether to add mem tokens at the end of sequence.')
parser.add_argument('--read_mem_from_cache', action='store_true',
                    help='Mem tokens attend to their mem representations.')
parser.add_argument('--log_interval', type=int, default=200,
                    help='Log period in batches')
parser.add_argument('--eval_interval', type=int, default=8000,
                    help='Evaluation period in batches')
parser.add_argument('--answer_size', type=int, default=24,
                    help='How many last tokens in segment to use for loss.')

args = parser.parse_args()
args.tied = not args.not_tied

#########
if args.device_ids is not None:
    args.device_ids = [int(i) for i in args.device_ids]
    print(args.device_ids)

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'
assert args.batch_size % args.batch_chunk == 0

args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
logging = create_exp_dir(args.work_dir,
    scripts_to_save=['train.py', 'mem_transformer.py', 'train_synthetic.py'], debug=args.debug)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

# Validate `--fp16` option
if args.fp16:
    if not args.cuda:
        print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
        args.fp16 = False
    else:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

device = torch.device('cuda' if args.cuda else 'cpu')
if args.cuda:
    device = torch.device(args.device_ids[0] if args.device_ids is not None else 0)

###############################################################################
# Load data
###############################################################################
stack = False
tr_iter = data_loader('train', path=args.data, task_name=args.dataset, batch_size=args.batch_size,
                                    tgt_len=args.tgt_len, device=device, stack=stack)
va_iter = data_loader('val', path=args.data, task_name=args.dataset, batch_size=args.batch_size,
                                    tgt_len=args.tgt_len, device=device, stack=stack)
te_iter = data_loader('test', path=args.data, task_name=args.dataset, batch_size=args.batch_size,
                                    tgt_len=args.tgt_len, device=device, stack=stack)
ntokens = args.ntokens = (tr_iter.src.max() + 1).item()

# # adaptive softmax / embedding
cutoffs, tie_projs = [], [False]

###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'n_emb_projs'):
            for i in range(m.n_emb_projs):
                if getattr(m, f'emb_projs_{i}') is not None:
                    nn.init.normal_(getattr(m, f'emb_projs_{i}'), 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'n_out_projs'):
            for i in range(m.n_out_projs):
                if getattr(m, f'out_projs_{i}') is not None:
                    nn.init.normal_(getattr(m, f'out_projs_{i}'), 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

if args.restart:
    with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    if not args.fp16:
        model = model.float()
    model.apply(update_dropout)
    model.apply(update_dropatt)
else:
    model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
        args.d_head, args.d_inner, args.dropout, args.dropatt,
        tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
        tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
        ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
        num_mem_tokens=args.num_mem_tokens, mem_at_end=args.mem_at_end, read_mem_from_cache=args.read_mem_from_cache, 
        same_length=args.same_length, attn_type=args.attn_type,
        clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)
    model.apply(weights_init)
    model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing
args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

if args.fp16:
    model = model.half()

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                          model, dim=1, device_ids=args.device_ids, output_device=device)
    else:
        para_model = nn.DataParallel(model, device_ids=args.device_ids, dim=1, output_device=device)

    para_model.num_mem_tokens = para_model.module.num_mem_tokens
else:
    para_model = model.to(device)

#### optimizer
if args.optim.lower() == 'sgd':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
        optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.mom)
elif args.optim.lower() == 'adam':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
        optimizer = optim.Adam(dense_params, lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optim.lower() == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

#### scheduler
if args.scheduler == 'cosine':
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        args.max_step, eta_min=args.eta_min) # should use eta_min arg
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
            args.max_step, eta_min=args.eta_min) # should use eta_min arg
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step \
                   else step / (args.warmup_step ** 1.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
            factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
elif args.scheduler == 'constant':
    pass

if args.cuda and args.fp16:
    # If args.dynamic_loss_scale is False, static_loss_scale will be used.
    # If args.dynamic_loss_scale is True, it will take precedence over static_loss_scale.
    optimizer = FP16_Optimizer(optimizer,
                               static_loss_scale = args.static_loss_scale,
                               dynamic_loss_scale = args.dynamic_loss_scale,
                               dynamic_loss_args = {'init_scale': 2 ** 16})

if args.restart:
    if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
        with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f)
            optimizer.load_state_dict(opt_state_dict)
    else:
        print('Optimizer was not saved. Start from scratch.')

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))

###############################################################################
# Training code
###############################################################################

def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    num_correct, num_correct_tf, num_total = 0, 0, 0
    num_correct_answers, num_total_answers = 0, 0
    with torch.no_grad():
        mems = tuple()  
        for i, (data_, target_, seq_len) in enumerate(eval_iter):
            if data_.shape[1] < args.batch_size:
                print('maslina')
                continue
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            mem_tokens = None
            if model.mem_tokens is not None:
                mem_tokens = model.mem_tokens.repeat(1, data_.shape[1], 1)

            data_segs = torch.chunk(data_, data_.shape[0] // args.tgt_len)
            target_segs = torch.chunk(target_, target_.shape[0] // args.tgt_len)
            losses = []
        
        # caclulate loss
            for data, target in zip(data_segs, target_segs):
                if mems is None:
                    mems = tuple()
                ret = para_model(data, target, *mems, mem_tokens=mem_tokens)
                if model.num_mem_tokens == 0:
                    loss, mems = ret[0], ret[1:]
                else:
                    mem_tokens, loss, mems = ret[0], ret[1], ret[2:]
                losses.append(loss)

            loss = torch.cat(losses)
            loss = loss[-args.answer_size:]
            loss = loss.mean()
            total_loss += args.answer_size * loss.float().item()
            total_len += args.answer_size

        # with teacher forcing
            mem_tokens = None
            if model.mem_tokens is not None:
                mem_tokens = model.mem_tokens.repeat(1, data_.shape[1], 1)

            pred_segs = []
            mems = tuple()    
            for data, target in zip(data_segs, target_segs):
                if mems is None:
                    mems = tuple()
                if not mems: mems = model.init_mems(data.device)

                tgt_len = target.size(0)
                hidden, mems = model._forward(data, mems=mems, mem_tokens=mem_tokens)
                num_mem = model.num_mem_tokens
                if model.num_mem_tokens > 0:
                    if model.mem_at_end:
                        pred_hid = hidden[-tgt_len - num_mem:-num_mem]
                        mem_tokens = hidden[-num_mem:]
                    else:
                        pred_hid = hidden[-tgt_len:]
                        mem_tokens = hidden[-tgt_len-num_mem:-tgt_len]
                else:
                    pred_hid = hidden[-tgt_len:]

                logit = model.crit._compute_logit(pred_hid, model.crit.out_layers[0].weight,
                                        model.crit.out_layers[0].bias, model.crit.out_projs_0)
                logit = torch.nn.functional.softmax(logit, dim=-1)
                preds = logit.argmax(dim=-1)
                pred_segs.append(preds)

            preds = torch.cat(pred_segs)
            num_total += (target_[-args.answer_size:] > 0).float().sum().item()
            num_correct_tf += ((preds[-args.answer_size:] == target_[-args.answer_size:]) & (target_[-args.answer_size:] > 0)).float().sum().item()
            
            S, T, P = data_[:, -1].cpu().numpy(), target_[:, -1].cpu().numpy(), preds.cpu().numpy()

        # no teacher forcing                
            mem_tokens, tmp_mem_tokens = None, None
            if model.mem_tokens is not None:
                mem_tokens = model.mem_tokens.repeat(1, data_.shape[1], 1)
            
            if args.answer_size >= args.tgt_len:
                q_data, q_target = data_[:-args.answer_size].clone(), target_[:-args.answer_size].clone()
                a_data, a_target = data_[-args.answer_size:].clone(), target_[-args.answer_size:].clone()

                q_chunks = q_data.shape[0] // args.tgt_len
                a_chunks = max((a_data.shape[0] // args.tgt_len, 1))
                q_data_segs = torch.chunk(q_data, q_chunks)
                q_target_segs = torch.chunk(q_target, q_chunks)
                a_data_segs = torch.chunk(a_data, a_chunks)
                a_target_segs = torch.chunk(a_target, a_chunks)
                
                mems, tmp_mems = tuple(), tuple()
                for data, target in zip(q_data_segs, q_target_segs):
                    if mems is None:
                        mems = tuple()
                    if not mems: mems = model.init_mems(data.device)

                    tgt_len = target.size(0)
                    hidden, mems = model._forward(data, mems=mems, mem_tokens=mem_tokens)
                    num_mem = model.num_mem_tokens
                    if model.num_mem_tokens > 0:
                        if model.mem_at_end:
                            pred_hid = hidden[-tgt_len - num_mem:-num_mem]
                            mem_tokens = hidden[-num_mem:]
                        else:
                            pred_hid = hidden[-tgt_len:]
                            mem_tokens = hidden[-tgt_len-num_mem:-tgt_len]
                
                target_preds = list(q_target_segs)
                start_ind = 0
            elif data_.shape[0] != args.tgt_len:
                print(f'Tgt len {args.tgt_len} but data shape {data_.shape}!')
                raise(NotImplementedError)
            else:
                a_data, a_target = data_.clone(), target_.clone()
                a_chunks = 1
                a_data_segs = torch.chunk(a_data, a_chunks)
                a_target_segs = torch.chunk(a_target, a_chunks)
                target_preds = list()
                start_ind = 30
            
            for data, target in zip(a_data_segs, a_target_segs):
                for token_ind in range(start_ind, args.tgt_len):
                    if mems is None:
                        mems = tuple()
                    if not mems: mems = model.init_mems(data.device)

                    tgt_len = target.size(0)
                    tmp_mems = mems
                    hidden, tmp_mems = model._forward(data, mems=tmp_mems, mem_tokens=mem_tokens)
                    num_mem = model.num_mem_tokens
                    if model.num_mem_tokens > 0:
                        if model.mem_at_end:
                            pred_hid = hidden[-tgt_len - num_mem:-num_mem]
                            tmp_mem_tokens = hidden[-num_mem:]
                        else:
                            pred_hid = hidden[-tgt_len:]
                            tmp_mem_tokens = hidden[-tgt_len-num_mem:-tgt_len]
                    else:
                        pred_hid = hidden[-tgt_len:]

                    logit = model.crit._compute_logit(pred_hid, model.crit.out_layers[0].weight,
                                            model.crit.out_layers[0].bias, model.crit.out_projs_0)
                    logit = torch.nn.functional.softmax(logit[token_ind], dim=-1)
                    preds = logit.argmax(dim=1)
                    
                    target[token_ind] = preds
                    if token_ind < args.tgt_len - 1:
                        data[token_ind + 1] = preds

                mems = tmp_mems
                mem_tokens = tmp_mem_tokens
                target_preds.append(target)

            target_preds = torch.cat(target_preds)
            correct = ((target_preds[-args.answer_size:] == target_[-args.answer_size:]) & (target_[-args.answer_size:] > 0)).sum().item()
            num_correct += correct
        # else:
            # target_preds = preds

        # answer accuracy
            for bi in range(args.batch_size):
                if (target_preds[-30:, bi] == target_[-30:, bi]).float().mean().item() == 1:
                    num_correct_answers += 1

            num_total_answers += args.batch_size

    logging(f'|\nSource: {S}\nTarget: {T}\nTeacher forcing: acc:{num_correct_tf/num_total}\nPreds:  {P[:, -1]}\n')
    logging(f'No teacher forcing: acc:{num_correct/num_total}\nPreds:  {target_preds[:, -1].cpu().numpy()}\n')
    accuracy = num_correct / num_total
    
    logging(f'Answer acc:{num_correct_answers/num_total_answers}\n\n')
        
    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()
    return total_loss / total_len, accuracy


def train():
    # Turn on training mode which enables dropout.
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    mem_tokens = None
    prev_data, prev_target, prev_mems, prev_mem_tokens = [], [], [], []
    for batch, (data_, target_, seq_len) in enumerate(train_iter):
        model.zero_grad()
         
        if args.batch_chunk > 1:
            raise(NotImplementedError)
        else:
            data_segs = torch.chunk(data_, data_.shape[0] // args.tgt_len)
            target_segs = torch.chunk(target_, target_.shape[0] // args.tgt_len)
            losses = []
            if model.mem_tokens is not None:
                mem_tokens = model.mem_tokens.repeat(1, data_.shape[1], 1)
            
            prev_data, prev_target, prev_mems, prev_mem_tokens = [], [], [], []
            for data, target in zip(data_segs, target_segs):
                if args.mem_backprop_depth > 0:
                    prev_data = prev_data[-args.mem_backprop_depth:] + [data]
                    prev_target = prev_target[-args.mem_backprop_depth:] + [target]
                    prev_mems = prev_mems[-args.mem_backprop_depth:] + [mems]
                    prev_mem_tokens = prev_mem_tokens[-args.mem_backprop_depth:] + [mem_tokens.detach()]
                    mem_tokens = prev_mem_tokens[0]
                    for pd, pt, pm in zip(prev_data[:-1], prev_target[:-1], prev_mems[:-1]):
                        ret = para_model(pd, pt, *pm, mem_tokens=mem_tokens)
                        if model.num_mem_tokens == 0:
                            loss, mems = ret[0], ret[1:]
                        else:
                            mem_tokens, loss, mems = ret[0], ret[1], ret[2:]
                        if args.bptt_bp:
                            raise(NotImplementedError)
                
                ret = para_model(data, target, *mems, mem_tokens=mem_tokens)
                if model.num_mem_tokens == 0:
                    loss, mems = ret[0], ret[1:]
                else:
                    mem_tokens, loss, mems = ret[0], ret[1], ret[2:]
                losses.append(loss)

            loss = torch.cat(losses)
            loss = loss[-args.answer_size:]
            loss = loss.float().mean().type_as(loss)
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            train_loss += loss.float().item()
            
        if args.fp16:
            optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()
        
        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)

        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.5f} | loss {:5.5f}'.format(
                epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
            else:
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
            logging(log_str)
            train_loss = 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            val_loss, val_acc = evaluate(va_iter)
            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.5f}'.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), val_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
            else:
                log_str += ' | valid ppl {:9.4f}'.format(math.exp(val_loss))
            log_str += ' | valid acc {}'.format(round(val_acc, 3))
            logging(log_str)
            logging('-' * 100)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug:
                    with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                        torch.save(model, f)
                    with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss = val_loss

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)
                if args.sample_softmax > 0:
                    scheduler_sparse.step(val_loss)

            eval_start_time = time.time()

        if train_step == args.max_step:
            break

# Loop over epochs.
train_step = 0
train_loss = 0
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        train()
        if train_step == args.max_step:
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
para_model = model.to(device)

# Run on test data.
test_loss, test_acc = evaluate(te_iter)
logging('=' * 100)
if args.dataset in ['enwik8', 'text8']:
    logging('| End of training | test loss {:5.5f} | test bpc {:9.5f}'.format(
        test_loss, test_loss / math.log(2)))
else:
    logging('| End of training | test loss {:5.5f} | test ppl {:9.5f}'.format(
        test_loss, math.exp(test_loss)))
logging(' | test acc {}'.format(round(test_acc, 3)))
logging('=' * 100)