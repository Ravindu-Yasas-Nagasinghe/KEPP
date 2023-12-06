import torch
from utils.args import get_args
args = get_args()
class LLMLabelOnehot(torch.nn.Module):
    def __init__(self, batch_size, T, num_lists, probabilities):
        super(LLMLabelOnehot, self).__init__()

        self.batch_size = batch_size
        self.T = T
        self.num_lists = num_lists
        self.probabilities = probabilities

        self.onehot = torch.zeros((batch_size * T, args.class_dim_llama)).cuda()

    def forward(self, LLM_label):
        for batch_idx in range(self.batch_size):
            for t in range(self.T):
                list_prob = torch.zeros(args.class_dim_llama).cuda()
                for list_idx in range(self.num_lists):
                    if self.probabilities==[1]:
                        list_prob[LLM_label[batch_idx][t]] = self.probabilities[list_idx]
                    else:
                        list_prob[LLM_label[batch_idx][list_idx][t]] = self.probabilities[list_idx]
                    
                self.onehot[batch_idx * self.T + t, :] = list_prob

        self.onehot = self.onehot.reshape(self.batch_size, self.T, -1).cuda()
        return self.onehot
    

class PKGLabelOnehot(torch.nn.Module):
    def __init__(self, batch_size, T, num_lists, probabilities):
        super(PKGLabelOnehot, self).__init__()

        self.batch_size = batch_size
        self.T = T
        self.num_lists = num_lists
        self.probabilities = probabilities

        self.onehot = torch.zeros((batch_size * T, args.class_dim_graph)).cuda()

    def forward(self, PKG_label):
        for batch_idx in range(self.batch_size):
            for t in range(self.T):
                list_prob = torch.zeros(args.class_dim_graph).cuda()
                for list_idx in range(self.num_lists):
                    if self.probabilities==[1]:
                        list_prob[PKG_label[batch_idx][t]] = self.probabilities[list_idx]
                    else:
                        list_prob[PKG_label[batch_idx][list_idx][t]] = self.probabilities[list_idx]
                    
                self.onehot[batch_idx * self.T + t, :] = list_prob

        self.onehot = self.onehot.reshape(self.batch_size, self.T, -1).cuda()
        return self.onehot