from .retrieval import *
# from remote.listops import *
import numpy as np
import torch


def generate_data(generator, task_name, path='data', train_size=10_000, val_size=1_000, test_size=2_000, batch_size=32):
    Xs, ys = [], []
    total_size = train_size + test_size + val_size
    num_batches = total_size // batch_size * 3
    for _ in range(num_batches):
        X, y, _, _ = next(generator)
        Xs.append(X)
        ys.append(y)

    Xs = torch.vstack(Xs)
    ys = torch.vstack(ys)
    
    _, inds = np.unique(Xs, axis=0, return_index=True)
    inds = np.random.permutation(inds)
    
    Xs = Xs[inds][:total_size]
    ys = ys[inds][:total_size]
    
    np.save(f'{path}/{task_name}_train_X.npy', Xs[:train_size].cpu())
    np.save(f'{path}/{task_name}_train_y.npy', ys[:train_size].cpu())

    np.save(f'{path}/{task_name}_val_X.npy', Xs[train_size:train_size+val_size].cpu())
    np.save(f'{path}/{task_name}_val_y.npy', ys[train_size:train_size+val_size].cpu())

    np.save(f'{path}/{task_name}_test_X.npy', Xs[train_size+val_size:train_size+val_size+test_size].cpu())
    np.save(f'{path}/{task_name}_test_y.npy', ys[train_size+val_size:train_size+val_size+test_size].cpu())


class data_loader:
    def __init__(self, mode, task_name, path='data', batch_size=32, tgt_len=24, device='cpu', stack=True):
        X = np.load(f'{path}/{task_name}_{mode}_X.npy')
        y = np.load(f'{path}/{task_name}_{mode}_y.npy')

        if stack:
            src = np.hstack((X, y))
            src, tgt = src[:, :-1], src[:, 1:]
        else:
            src, tgt = X, y

        self.src = torch.Tensor(src).long()
        self.tgt = torch.Tensor(tgt).long()

        self.data_size = self.src.shape[0]
        self.data_ptr = 0

        self.batch_size = batch_size
        self.tgt_len = tgt_len
        self.device = device

    def __iter__(self):
        return next(self)

    def __next__(self):
        while(True):
            src = self.src[self.data_ptr: self.data_ptr+self.batch_size].to(self.device)
            tgt = self.tgt[self.data_ptr: self.data_ptr+self.batch_size].to(self.device)
            
            self.data_ptr = (self.data_ptr + self.batch_size) % self.data_size

            yield src.T, tgt.T, self.tgt_len
        
    
class copy_generator:
    def __init__(self, batch_size, enc_seq_len, dec_seq_len, num_tokens):
        self.src_mask = torch.ones(batch_size, enc_seq_len).bool().cuda()
        self.tgt_mask = torch.ones(batch_size, dec_seq_len+1).bool().cuda()
    
        self.batch_size = batch_size
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.num_tokens = num_tokens

    def __next__(self):
        X = np.zeros([self.batch_size, self.enc_seq_len]).astype(int)
        y = np.zeros([self.batch_size, self.dec_seq_len+1]).astype(int)
        y[:, 0] = 1
        for i in range(self.batch_size):
            sequence_length = self.enc_seq_len
            random_sequence = np.random.randint(2, self.num_tokens, sequence_length)
            
            X[i, :sequence_length] = random_sequence
            y[i, 1: 2 * sequence_length + 1] = np.concatenate([random_sequence] * 2)

        return torch.tensor(X), torch.tensor(y), self.src_mask, self.tgt_mask        


class reverse_generator:
    def __init__(self, batch_size, enc_seq_len, dec_seq_len, num_tokens):
        self.src_mask = torch.ones(batch_size, enc_seq_len).bool().cuda()
        self.tgt_mask = torch.ones(batch_size, dec_seq_len+1).bool().cuda()
        
        self.batch_size = batch_size
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.num_tokens = num_tokens
    
    def __next__(self):
        X = np.zeros([self.batch_size, self.enc_seq_len]).astype(int)
        y = np.zeros([self.batch_size, self.dec_seq_len+1]).astype(int)
        y[:, 0] = 1
        for i in range(self.batch_size):
            sequence_length = self.enc_seq_len
            random_sequence = np.random.randint(2, self.num_tokens, sequence_length)
            
            X[i, :sequence_length] = random_sequence
            y[i, 1:sequence_length + 1] = random_sequence[::-1]


        return torch.tensor(X), torch.tensor(y), self.src_mask, self.tgt_mask        


class retrieval_generator:
    def __init__(self, K=4, num_tokens=10, batch_size=128):
        self.src_mask = torch.ones(batch_size, 2*K+1).bool()
        self.tgt_mask = torch.ones(batch_size, 1+1).bool()
        self.K = K
        self.batch_size = batch_size
        self.N = num_tokens
    
    def __next__(self):
        X = np.zeros([self.batch_size, 2*self.K+1]).astype(int)
        y = np.zeros([self.batch_size, 1+1]).astype(int)
        y[:, 0] = 10
        for i in range(self.batch_size):
            X[i], y[i, 1:] = create_sequence(one_hot=False, K=self.K)


        return torch.tensor(X), torch.tensor(y), self.src_mask, self.tgt_mask         


# class listops_generator:
#     def __init__(self, max_depth=2):
#         self.src_mask = torch.ones(BATCH_SIZE, enc_seq_len).bool().cuda()
#         self.tgt_mask = torch.ones(BATCH_SIZE, dec_seq_len+1).bool().cuda()
#         self.max_depth = max_depth
    
#     def __next__(self):
#         X = np.zeros([BATCH_SIZE, enc_seq_len]).astype(int)
#         y = np.ones([BATCH_SIZE, 2]).astype(int) * 2
#         for i in range(BATCH_SIZE):
#             t = generate_tree(self.max_depth)
#             tokens, value = to_tokens(t), to_value(t) 
#             X[i, 0:len(tokens)], y[i, 1:] = tokens, value+2
#             del t

#         return torch.tensor(X), torch.tensor(y), self.src_mask, self.tgt_mask         


# if __name__ == "__main__":
#     task2gen = {'copy': copy_generator,
#                 'reverse': reverse_generator,
#                 'retrieval': retrieval_generator,
#                 'listops': listops_generator}

#     generator = task2gen[TASK_NAME]()
#     generate_data(generator, task_name=TASK_NAME, train_size=TRAIN_SIZE, test_size=TEST_SIZE, val_size=VAL_SIZE)  