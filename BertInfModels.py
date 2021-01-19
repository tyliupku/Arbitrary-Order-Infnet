import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class tagrnn(nn.Module):
    def __init__(self, num_tags, RNN_TYPE="LSTM", NUM_DIRS=2, taglm_hidden_size=100,
                 taglm_num_dir=2, taglm_num_layer=1, taglm_worddp=0, taglm_rnndp=0):
        super().__init__()
        self.batch_size = 0

        # architecture
        self.rnn = getattr(nn, RNN_TYPE)(
                input_size=num_tags,
                hidden_size=taglm_hidden_size // NUM_DIRS,
                num_layers=taglm_num_layer,
                bias=True,
                batch_first=True,
                dropout=taglm_rnndp,
                bidirectional=(taglm_num_dir == 2)
        )
        self.sos = nn.Parameter(torch.FloatTensor())
        self.num_layers = taglm_num_layer
        self.num_tags = num_tags
        self.dropout = nn.Dropout(taglm_worddp)
        self.sm = nn.Softmax(dim=2)
        self.logsm = nn.LogSoftmax(dim=2)
        self.eps = 1e-9
        self.out = nn.Linear(taglm_hidden_size, num_tags)  # RNN output to tag
        self = self.cuda()

    def forward(self, y_in, mask, activate="logsm"):
        # y_in: [B, T, C]
        y_in = self.dropout(y_in)
        # x = nn.utils.rnn.pack_padded_sequence(y_in, mask.sum(1).int(), batch_first=True)
        h, _ = self.rnn(y_in)
        # h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h = self.out(h)
        if activate=="sm":
            h = self.sm(h)
        elif activate=="logsm":
            h = self.logsm(h)
        h = h * mask.float().unsqueeze(2)
        return h


class cnn(nn.Module):
    def __init__(self, num_tags, cnn_kernel_num=50, cnn_kernel_size=[3], cnn_dp=0):
        super(cnn, self).__init__()

        D = num_tags
        C = num_tags
        Ci = 1
        Co = cnn_kernel_num
        Ks = cnn_kernel_size

        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), padding=(int(K/2),0)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(cnn_dp)
        # self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, mask):
        # x: [B, T, C]  mask: [B, T]
        x = x * mask.unsqueeze(2)
        x = x.unsqueeze(1)  # (N, Ci, W, D) (B, Ci, T, C)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(B, Co, T), ...]*len(Ks)
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)  # [B, len(Ks)*Co, T]

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (B, len(Ks)*Co, T)
        return x.sum(1)  # [B, T]

    def vis(self, x, mask):
        # x: [B, T, C]  mask: [B, T]
        x = x * mask.unsqueeze(2)
        x = x.unsqueeze(1)  # (N, Ci, W, D) (B, Ci, T, C)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(B, Co, T), ...]*len(Ks)
        x = torch.cat(x, 1)  # [B, len(Ks)*Co, T]

        return x.transpose(1,2)  # [B, T, Co]


class selfatt(nn.Module):
    def __init__(self, num_tags, selfatt_form="max"):
        super(selfatt, self).__init__()
        self.scale = 1. / math.sqrt(num_tags)
        self.form = selfatt_form

    def forward(self, query, keys, values, mask, m=0):
        # Query = [B, T, C]
        # Keys = [B, T, C]
        # Values = [B, T, C]
        # Outputs = [B]

        B, T, C = query.size()
        keys = keys.transpose(1,2) # [B, C, T]
        # keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys)  # [B, T, C]x[B, C, T] -> [B, T, T]
        if m>0:
            selfatt_mask = torch.cuda.FloatTensor(T, T).fill_(-1e10)  # [T, T]
            for i in range(T):
                lower_idx, higher_idx = max(0, i-m), min(T, i+m+1)
                selfatt_mask[i][lower_idx:higher_idx] = 0
            energy = energy + selfatt_mask.unsqueeze(0)  # [B, T, T]
        energy = F.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize

        linear_combination = torch.bmm(energy, values) # [B, T, T]x[B, T, C] -> [B, T, C]
        linear_combination = linear_combination * mask.unsqueeze(2) # [B, T, C]
        if self.form == "max":
            return linear_combination.max(dim=2)[0].sum(1) # [B, T, C] -> [B]
        elif self.form == "sum":
            return linear_combination.sum(2).sum(1) # [B, T, C] -> [B]


class BERT_infnet(nn.Module):
    def __init__(self, localnet, infnet, num_tags, args, start_label_id, stop_label_id, margin_type=0, regu_type=0, M=1):
        super(BERT_infnet, self).__init__()
        self.localnet = localnet
        self.infnet = infnet
        self.cost_infnet = nn.Sequential(nn.Linear(2 * num_tags, num_tags), nn.Softmax(dim=2))

        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_tags = num_tags
        # self.max_seq_length = max_seq_length
        self.batch_size = args.batch_size

        self.dropout = torch.nn.Dropout(0.2)
        # Maps the output of the bert into label space.
        self.local2label = nn.Linear(self.hidden_size, self.num_tags)
        self.inf2label = nn.Linear(self.hidden_size, self.num_tags)

        self.M = args.M
        if args.Wyy_form == "direct":
            self.Wyy = nn.Parameter(
                torch.FloatTensor(np.random.uniform(-0.02, 0.02, (self.M, num_tags + 1, num_tags + 1)).astype('float32')))
        elif args.Wyy_form == "horderm":
            self.Wyy = nn.Parameter(
                torch.FloatTensor(np.random.uniform(-0.02, 0.02, [num_tags+1] * (self.M+1)).astype('float32')))
        elif args.Wyy_form == "horderresid":
            self.wembs = nn.Parameter(torch.Tensor(self.num_tags+1, args.Wyy_form.howemb_size))
            self.wdecoder = nn.Linear(self.M*args.Wyy_form.howemb_size, self.num_tags+1)
            self.wmlp = nn.Sequential(nn.Linear(self.M*args.Wyy_form.howemb_size, self.M*args.Wyy_form.howemb_size),
                                           nn.ReLU(), nn.Dropout(args.Wyy_form.howdropout))
            self.wnorm = nn.LayerNorm(self.M*args.Wyy_form.howemb_size)
        elif args.Wyy_form == "decom":
            self.Es = nn.ModuleList([nn.Linear(num_tags, args.wdecom_size)] * (self.M+1))
        elif "taglm" in args.Wyy_form:
            self.tag_rnn = tagrnn(self.num_tags)
            self.eps = 1e-10
        elif args.Wyy_form == "selfatt":
            self.tag_selfatt = selfatt(self.num_tags)
        elif args.Wyy_form == "cnn":
            self.tag_cnn = cnn(self.num_tags)
        else:
            self.wembs = nn.Parameter(torch.Tensor(self.M*(num_tags+1), args.wemb_size))
            wdrop = nn.Dropout(args.wdropout)
            if args.Wyy_form == "mlp":
                self.decoder = nn.Sequential(nn.Linear(args.wemb_size, args.whid_size), nn.Tanh(),
                                             wdrop, nn.Linear(args.whid_size, num_tags+1))
            elif args.Wyy_form == "resid":
                self.mlp = nn.Sequential(nn.Linear(args.wemb_size, args.wemb_size),
                                         nn.Tanh(), wdrop)
                self.decoder = nn.Linear(args.wemb_size, num_tags+1)
                self.norm = nn.LayerNorm(args.wemb_size)
        self.eps = 1e-9
        self.margin_type = margin_type
        self.regu_type = regu_type
        self.args = args
        self = self.cuda()

    def get_trans_matrix(self):
        # y_in : [B, T, C]
        if self.args.Wyy_form == "direct":
            return self.Wyy
        elif self.args.Wyy_form == "mlp":
            return self.decoder(self.wembs).reshape((self.M, self.num_tags+1, self.num_tags+1))
        elif self.args.Wyy_form == "resid":
            return self.decoder(self.norm(self.wembs+self.mlp(self.wembs))).reshape((self.M, self.num_tags+1, self.num_tags+1))
        elif self.args.Wyy_form == "decom":
            # Wyy = torch.mm(self.E1.weight.transpose(0,1), self.E2.weight)
            Wyy0 = self.Es[0].weight.transpose(0, 1)
            Wyys = []
            for i in range(1, self.M+1):
                Wyys.append(torch.mm(Wyy0, self.Es[i].weight)[None, :, :])
            Wyys = torch.cat(Wyys, dim=0)
            # print("Wyys shape {}".format(Wyys.size()))
            return Wyys
        elif self.args.Wyy_form == "horderm":
            return self.Wyy
        elif self.args.Wyy_form == "horderresid":
            cat_labe_embs = []
            Kp1 = self.num_tags+1 # extra class for start state
            # make cartesian product of label embeddings
            for m in range(self.M):
                nreps = Kp1**(self.M-m-1) # number of times to repeat each label embedding
                # make block of size (K+1)^M-m x lemb_size
                block = self.wembs.unsqueeze(1).repeat(1, nreps, 1).view(-1, self.wembs.size(1))
                breps = Kp1**m # number of times to repeat each block
                cat_labe_embs.append(block.repeat(breps, 1))
            cat_labe_embs = torch.cat(cat_labe_embs, 1) # K+1^M x M*lemb_size
            assert cat_labe_embs.size(0) == Kp1**self.M
            # print(cat_labe_embs.size())
            # print(cat_labe_embs[:, 2])
            tscores = self.wdecoder( # K+1^(M+1)
                self.wnorm(cat_labe_embs + self.wmlp(cat_labe_embs)))
            # print(tscores.size())
            tdims = [Kp1]*(self.M+1)
            return F.softmax(tscores, dim=1).view(*tdims)
        else: return None

    def get_trans_energy(self, Wyy, y_in, mask, B, T, C):
        if self.args.Wyy_form == "decom":
            # y_in: [B, T, C]  mask: [B, T]
            # y_in: y0 y1 y2 y3 y4
            # when M=4 fully connected energy
            # when i=1
            # y_in[:, :-1]: y0 y1 y2 y3
            # y_in[:, 1:] : y1 y2 y3 y4
            #
            # when i=2
            # y_in[:, :-2]: y0 y1 y2
            # y_in[:, 2:] : y2 y3 y4
            #
            # when i=3
            # y_in[:, :-3]: y0 y1
            # y_in[:, 3:] : y3 y4
            #
            # when i=4
            # y_in[:, :-3]: y0
            # y_in[:, 3:] : y4
            trans_energy = 0
            for i in range(1, min(self.M+1, T+1)):
                trans_energy += ((self.Es[0](y_in[:, :-i]) * self.Es[i](y_in[:, i:])).sum(2) * mask[:, i:]).sum(1)
            return trans_energy
        elif "taglm" in self.args.Wyy_form:
            trans_energy = 0
            if self.args.Wyy_form == "taglm-log":
                y_lm = self.tag_rnn(y_in, mask, activate="sm")
                trans_energy = torch.log((y_lm * y_in + self.eps).sum(2)).sum(1)
            elif self.args.Wyy_form == "taglm-ylog":
                y_lm = self.tag_rnn(y_in, mask, activate="logsm")
                trans_energy = ((y_lm * y_in).sum(2)).sum(1)
            elif self.args.Wyy_form == "taglm-log-seg":
                for t in range(1, T):
                    y_lm = self.tag_rnn(y_in[:, max(0, t-self.M):t], mask[:, max(0, t-self.M):t], activate="sm")
                    trans_energy += torch.log((y_lm[:, -1] * y_in[:, t] + self.eps).sum(1))
            elif self.args.Wyy_form == "taglm-ylog-seg":
                for t in range(1, T):
                    y_lm = self.tag_rnn(y_in[:, max(0, t-self.M):t], mask[:, max(0, t-self.M):t], activate="logsm")
                    trans_energy += ((y_lm * y_in).sum(2)).sum(1)
            return trans_energy
        elif self.args.Wyy_form == "cnn":
            y_cnn = self.tag_cnn(y_in, mask)
            y_cnn = y_cnn * mask
            trans_energy = y_cnn.sum(1)
            return trans_energy
        elif self.args.Wyy_form == "selfatt":
            return self.tag_selfatt(y_in, y_in, y_in, mask, m=self.M)
        elif "horder" in self.args.Wyy_form:
            targets = y_in.transpose(0,1)
            Wtrans0 = _get_start_trans(Wyy, self.num_tags, self.M)  #[C+1]
            Wtrans0 = _get_trans_without_eos_sos(Wtrans0, self.num_tags, 0)  #[C]
            energy_0 = torch.mm(targets[0], Wtrans0.unsqueeze(1)).squeeze(1)  #[B]
            trans_energy = energy_0

            trans_no_eos = _get_trans_without_eos_sos(Wyy, self.num_tags, self.M)  # [C]*(M+1) transition matrix without <sos> or <eos>
            # print("trans_no_eos {}".format(trans_no_eos.size()))
            # print("M {}".format(self.M))
            trans_no_eos = trans_no_eos.reshape((-1, C))  # [C^M, C]
            # print("trans_no_eos {}".format(trans_no_eos.size()))
            for t in range(1, T):
                energy_t = 0
                target_t = targets[t]  # [B, C]
                if t < self.M:
                    Wtrans = _get_start_trans(Wyy, self.num_tags, self.M - t)  # [C+1]*(t+1)
                    Wtrans = _get_trans_without_eos_sos(Wtrans, self.num_tags, t)  # [C]*(t+1)
                    Wtrans = Wtrans.reshape((-1, C)) # [C^t, C]
                    com_targets = targets[0]  # [B, C]
                    for i in range(1, t):
                        # [B, C^i, 1] x [B, 1, C] -> [B, C^i, C]
                        com_targets = torch.matmul(com_targets.unsqueeze(2), targets[i].unsqueeze(1))
                        com_targets = com_targets.reshape((B, -1))  # [B, C^(i+1)]
                    new_ta_energy = torch.mm(com_targets, Wtrans)  # [B, C^t] x [C^t, C] -> [B, C]
                    energy_t += ((new_ta_energy * target_t).sum(1)) * mask[:, t]  # [B]
                else:
                    com_targets = targets[t-self.M] # [B, C]
                    for i in range(t-self.M+1, t):
                        # [B, C^i, 1] x [B, 1, C] -> [B, C^i, C]
                        com_targets = torch.matmul(com_targets.unsqueeze(2), targets[i].unsqueeze(1))
                        com_targets = com_targets.reshape((B, -1))  # [B, C^(i+1)]
                    # print("com_targets : {}".format(com_targets.size()))
                    new_ta_energy = torch.mm(com_targets, trans_no_eos)  # [B, C^M] x [C^M, C] -> [B, C]
                    # print("new_ta_energy : {}".format(new_ta_energy.size()))
                    energy_t += ((new_ta_energy * target_t).sum(1)) * mask[:, t]  # [B]
                trans_energy += energy_t
            '''
            # ignore -> <eos> energy
            for i in range(min(self.M, T)):
                Wtrans = _get_end_trans(self.Wyy, self.num_tags, i+1, self.M)  #[C+1]*(M-i)
                Wtrans = _get_trans_without_eos_sos(Wtrans, self.num_tags, self.M-i)  #[C]*(M-i)
            '''
            return trans_energy
        else:
            targets = y_in.transpose(0, 1)  # [T, B, C]
            length_index = mask.sum(1).long() - 1  # [B]

            trans_energy = 0
            prev_labels = []
            for t in range(T):
                energy_t = 0
                target_t = targets[t]  # [B, C]
                if t < self.M:
                    prev_labels.append(target_t)
                    new_ta_energy = torch.mm(prev_labels[t], Wyy[t, -1, :-1].unsqueeze(1)).squeeze(1)  # [B, C] x [C] -> [B]
                    energy_t += new_ta_energy * mask[:, t]  # [B]
                    for i in range(t):
                        new_ta_energy = torch.mm(prev_labels[t-1-i], Wyy[i, :-1, :-1])  # [B, C]
                        energy_t += ((new_ta_energy * target_t).sum(1)) * mask[:, t]  # [B]
                else:
                    for i in range(self.M):
                        new_ta_energy = torch.mm(prev_labels[self.M-1-i], Wyy[i, :-1, :-1])  # [B, C]
                        energy_t += ((new_ta_energy * target_t).sum(1)) * mask[:, t]  # [B]
                    prev_labels.append(target_t)
                    prev_labels.pop(0)
                trans_energy += energy_t

            for i in range(min(self.M, T)):
                pos_end_target = y_in[torch.arange(B), length_index-i, :]  # [B, C]
                trans_energy += torch.mm(pos_end_target, Wyy[i, :-1, -1].unsqueeze(1)).squeeze(1) # [B]
            return trans_energy

    def forward(self, input_ids, segment_ids, input_mask, label_ids,
                epoch=0, return_energy=False):  # for training
        local_energy, _ = self.localnet(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                        output_all_encoded_layers=False)
        local_energy = self.dropout(local_energy)
        local_energy = self.local2label(local_energy)

        B, T, _ = local_energy.size()
        C = self.num_tags

        predy_inf, _ = self.infnet(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                output_all_encoded_layers=False)
        predy_inf = self.dropout(predy_inf)
        predy_inf = F.softmax(self.inf2label(predy_inf), dim=-1)

        y_in = F.one_hot(label_ids, num_classes=C).float()  # [B, T, C]
        predy = self.cost_infnet(torch.cat((predy_inf, y_in), 2))
        predy = predy * input_mask.unsqueeze(2)

        Wyy = self.get_trans_matrix()

        # print("predy_inf {} predy {}".format(predy_inf.size(), predy.size()))
        # print("Wyy {}".format(Wyy.size()))


        """for ground-truth energy"""
        trans_energy_pos = self.get_trans_energy(Wyy, y_in, input_mask, B, T, C)
        pos_local_energy = ((local_energy * y_in).sum(2) * input_mask).sum(1)  # [B, T, C] -> [B]
        pos_cost = pos_local_energy + trans_energy_pos

        """for cost-augmented InfNet"""
        trans_energy_neg = self.get_trans_energy(Wyy, predy, input_mask, B, T, C)
        neg_local_energy = ((local_energy * predy).sum(2) * input_mask).sum(1)  # [B, T, C] -> [B]
        neg_cost = neg_local_energy + trans_energy_neg

        """for InfNet"""
        trans_energy_inf = self.get_trans_energy(Wyy, predy_inf, input_mask, B, T, C)
        neg_inf_local_energy = ((local_energy * predy_inf).sum(2) * input_mask).sum(1)  # [B, T, C] -> [B]
        neg_inf_cost = neg_inf_local_energy + trans_energy_inf

        hinge_cost_inf = neg_inf_cost - pos_cost
        delta0 = (torch.abs(y_in - predy).sum(2) * input_mask).sum(1)  # [B]
        if self.margin_type == 1:
            hinge_cost0 = 1 + neg_cost - pos_cost
        elif self.margin_type == 2:
            hinge_cost0 = neg_cost - pos_cost
        elif self.margin_type == 0:
            hinge_cost0 = delta0 + neg_cost - pos_cost
        elif self.margin_type == 3:
            hinge_cost0 = delta0 * (1.0 + neg_cost - pos_cost)

        nllloss = nn.NLLLoss(reduction='none')
        predy_inf_f = predy_inf.view(-1, C)  # [B*T, C]
        logpredy_inf = torch.log(self.eps + predy_inf_f)  # [B*T, C]
        ce_hinge_inf = nllloss(logpredy_inf, label_ids.view(-1))
        ce_hinge_inf = (ce_hinge_inf.view(B, T) * input_mask).sum(1)  # [B]


        if self.regu_type == 0:
            margin, perceptron, celoss = torch.mean(-hinge_cost0), torch.mean(-hinge_cost_inf), torch.mean(ce_hinge_inf)
            g_cost = margin + self.args.Lambda * perceptron + self.args.CE_weight * np.exp(-self.args.zeta * epoch) * celoss
        else:
            g_cost = torch.mean(-hinge_cost0) + self.args.Lambda * torch.mean(-hinge_cost_inf)

        d_cost = torch.mean(F.relu(hinge_cost0)) + self.args.Lambda * torch.mean(F.relu(hinge_cost_inf))
        if return_energy:
            return (g_cost, margin, perceptron, celoss), d_cost, (-pos_local_energy.mean(), -trans_energy_pos.mean(),
                                                                  -neg_local_energy.mean(), -trans_energy_neg.mean(),
                                                                  -neg_inf_local_energy.mean(), -trans_energy_inf.mean())
        else:
            return (g_cost, margin, perceptron, celoss), d_cost

    def decode(self, input_ids, segment_ids, input_mask):
        predy_inf, _ = self.infnet(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                output_all_encoded_layers=False)
        predy_inf = F.softmax(self.inf2label(predy_inf), dim=-1)
        return predy_inf

def _get_end_trans(trans, K, M, maxM):
    assert M <= maxM
    if M == 1:
        if maxM == 1:
            slps = trans[:, K]
        elif maxM == 2:
            slps = trans[:, :, K]
        elif maxM == 3:
            slps = trans[:, :, :, K]
        elif maxM == 4:
            slps = trans[:, :, :, :, K]
    elif M == 2:
        slps = trans[K, K]
        if maxM == 2:
            slps = trans[:, K, K]
        elif maxM == 3:
            slps = trans[:, :, K, K]
        elif maxM == 4:
            slps = trans[:, :, :, K, K]
    elif M == 3:
        slps = trans[K, K, K]
        if maxM == 3:
            slps = trans[:, K, K, K]
        elif maxM == 4:
            slps = trans[:, :, K, K, K]
    elif M == 4:
        if maxM == 4:
            slps = trans[:, K, K, K, K]
    return slps


def _get_start_trans(trans, K, M):
    if M == 1:
        slps = trans[K]
    elif M == 2:
        slps = trans[K, K]
    elif M == 3:
        slps = trans[K, K, K]
    elif M == 4:
        slps = trans[K, K, K, K]
    return slps


def _get_trans_without_eos_sos(trans, K, M):
    if M == 0:
        tlps = trans[:K]
    elif M == 1:
        tlps = trans[:K, :K]
    elif M == 2:
        tlps = trans[:K, :K, :K]
    elif M == 3:
        tlps = trans[:K, :K, :K, :K]
    elif M == 4:
        tlps = trans[:K, :K, :K, :K, :K]
    return tlps