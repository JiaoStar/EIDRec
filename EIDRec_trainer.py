import numpy as np
from tqdm import tqdm
import random
import gc
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from EIDRec_models import KMeans
from utils import recall_at_k, ndcg_k


class Trainer:
    def __init__(self, model, train_dataloader,cluster_dataloader, eval_dataloader, test_dataloader, device, args):

        self.args = args
        self.device = device
        self.model = model
        self.seq_model = self.model.seq_model

        self.batch_size = self.args.batch_size
        self.sim=self.args.sim

        cluster = KMeans(
            num_cluster=args.intent_num,
            seed=1,
            hidden_size=args.hidden_size,
            device=device,
        )
        self.clusters = [cluster]
        self.clusters_t=[self.clusters]

        self.train_dataloader = train_dataloader
        self.cluster_dataloader=cluster_dataloader

        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.seq_model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader,self.cluster_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        raise NotImplementedError
    
    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HR@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HR@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def train_matrix_mask(self, user_ids):
        batch_user_index = user_ids.detach().cpu().numpy()
        mask = self.args.train_matrix[batch_user_index].toarray() > 0
        return torch.as_tensor(mask, dtype=torch.bool, device=self.device)

    
    @torch.no_grad()
    def get_sequences_in_same_cluster(self, input_sequence):
        input_sequence = input_sequence.to(self.device)
        batch_size = input_sequence.size(0)
        sequence_output = self.seq_model(input_sequence)
        sequence_representation = sequence_output[:, -1, :]  

        sequence_representation_np = sequence_representation.detach().cpu().numpy()
        cluster = self.clusters_t[0][0]  
        D, I = cluster.index.search(sequence_representation_np, 1)

        random_sequences = []
        for b in range(batch_size):
            cluster_idx = int(I[b][0])
            sequence_ids_in_cluster = cluster.cluster2sequences.get(cluster_idx, [])
            if len(sequence_ids_in_cluster) == 0:
                random_sequences.append(input_sequence[b].detach().cpu().numpy())
            else:
                seq_id = random.choice(sequence_ids_in_cluster)
                random_sequences.append(self.sequences_list[seq_id])

        return random_sequences

    @torch.no_grad()
    def make_random_reorder_view(self, input_sequence, input_seq_len):
        """Randomly reorder each user's own sequence, matching D_random semantics."""
        augmented = input_sequence.clone()
        max_len = input_sequence.size(1)

        for row_idx in range(input_sequence.size(0)):
            actual_length = input_seq_len[row_idx]
            if torch.is_tensor(actual_length):
                actual_length = actual_length.detach().cpu().item()
            actual_length = max(0, min(int(actual_length), max_len))
            if actual_length <= 1:
                continue

            start = max_len - actual_length
            valid_items = input_sequence[row_idx, start:].clone()
            perm = torch.randperm(actual_length, device=input_sequence.device)
            augmented[row_idx, start:] = valid_items[perm]

        return augmented
    

class EIDRecTrainer(Trainer):
    def __init__(self, model, train_dataloader,cluster_dataloader,eval_dataloader, test_dataloader, device, args):
        super(EIDRecTrainer, self).__init__(model, train_dataloader, cluster_dataloader,eval_dataloader, test_dataloader, device, args)

    def iteration(self, epoch, dataloader,cluster_dataloader=None, train=True):
        if train:
            cluster_interval = max(1, int(getattr(self.args, "cluster_update_interval", 1)))
            need_cluster = (not hasattr(self, "sequences_list")) or (epoch % cluster_interval == 0)
            if need_cluster:
                print("Preparing Clustering:")
                self.model.eval()
                kmeans_training_data = []
                sequence_ids = []
                sequences_list = []
                sequence_counter = 0

                rec_t_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
                with torch.no_grad():
                    for i, (rec_batch) in rec_t_data_iter:
                        rec_batch = tuple(t.to(self.device) for t in rec_batch)
                        user_id, subsequence, _, _, _ = rec_batch
                        batch_size = subsequence.size(0)

                        sequence_output_a = self.seq_model(subsequence)
                        sequence_output_b=sequence_output_a[:,-1,:]
                        kmeans_training_data.append(sequence_output_b.detach().cpu().numpy())
                        sequences_list.append(subsequence.detach().cpu().numpy())

                        batch_sequence_ids = np.arange(sequence_counter, sequence_counter + batch_size)
                        sequence_counter += batch_size
                        sequence_ids.extend(batch_sequence_ids)

                sequences_list = np.concatenate(sequences_list, axis=0)
                sequence_ids = np.array(sequence_ids)

                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
                kmeans_training_data_t = [kmeans_training_data]

                sequences_list_t = [sequences_list]
                sequence_ids_t = [sequence_ids]

                for i, clusters in enumerate(self.clusters_t):
                    for j, cluster in enumerate(clusters):
                        cluster.train(kmeans_training_data_t[i], sequence_ids_t[i])
                        self.clusters_t[i][j] = cluster

                self.sequences_list = sequences_list_t[0]

                # clean memory
                del kmeans_training_data
                del kmeans_training_data_t

                del sequences_list
                del sequences_list_t
                del sequence_ids
                del sequence_ids_t
                gc.collect()
            else:
                print(f"Reusing intent clusters from previous epoch (interval={cluster_interval}).")
            

            self.model.train()
            avg_loss = 0.0

            # print(f"rec dataset length: {len(dataloader)}")
            rec_t_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            # minibatch
            for i, (rec_batch) in rec_t_data_iter:

                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_seq, input_seq_len, target_pos, target = rec_batch
                
                same_cl_seqs =  self.get_sequences_in_same_cluster(input_seq)
                same_cl_seqs = np.array(same_cl_seqs)
                same_cl_seqs = torch.tensor(same_cl_seqs, dtype=torch.long).to(self.device)

                x0_gt = self.model.calculate_x0(target) # B x D
                
                s = self.model.calculate_guidance_s(same_cl_seqs, self.args.p, apply_dropout=True)

                t = torch.randint(0, self.args.timesteps, (input_seq.size(0), ), device=self.device).long()
                diff_loss, predicted_x_0 = self.calculate_diff_loss(self.model, x0_gt, s, t)
                
                # ablation
                if self.args.diff_weight == 0:
                    sequences_tensor = self.make_random_reorder_view(input_seq, input_seq_len)
                    x0_hat_out = self.seq_model(sequences_tensor)
                    x0_hat_out = x0_hat_out[:,-1,:]
                
                else:
                    sample_s = self.model.calculate_guidance_s(same_cl_seqs, self.args.p, apply_dropout=False)
                    x0_hat_out = self.model.sample_from_reverse_process(sample_s) # BxD

                x0_out = self.seq_model(input_seq) # BxLxD

                if self.args.spectral_view_filter:
                    x0_hat_out = self.model.filter_generated_view(
                        original_seq_hidden=x0_out,
                        generated_last_rep=x0_hat_out
                    )

                cl_loss = self.calculate_cl_loss(x0_hat_out, x0_out[:,-1,:], target_pos[:, -1])

                logits = self.predict_full(x0_out[:, -1, :])  #  Bx|I| 
                rec_loss = nn.CrossEntropyLoss()(logits, target_pos[:, -1])

                multi_task_loss = self.args.rec_weight*rec_loss + self.args.diff_weight*diff_loss + self.args.cl_weight*cl_loss
                

                self.optimizer.zero_grad()
                multi_task_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                avg_loss += rec_loss.item()
                
            log_record = {
                "epoch": epoch,
                "avg_loss": "{:.4f}".format(avg_loss / len(rec_t_data_iter)),
                "spectral_view_filter": self.args.spectral_view_filter,
                "spectral_tau_init": self.args.spectral_tau_init,
                "spectral_lambda_scale": self.args.spectral_lambda_scale,
                "spectral_filter_residual": self.args.spectral_filter_residual,
                "guide_mode": self.args.guide_mode,
                "guide_entropy_lambda": self.args.guide_entropy_lambda,
                "guide_entropy_temperature": self.args.guide_entropy_temperature,
                "sample_steps": getattr(self.args, "sample_steps", 0),
                "sample_eta": getattr(self.args, "sample_eta", 1.0),
                "cluster_update_interval": getattr(self.args, "cluster_update_interval", 1),
            }

            if (epoch+1) % self.args.print_log_freq == 0:
                print(str(log_record))

            with open(self.args.log_file, "a") as f:
                f.write(str(log_record) + "\n")

        else: # Val or Test
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader)) 
            self.model.eval()
            pred_list, answer_list = None, None


            pred_batches, answer_batches = [], []
            with torch.inference_mode():
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids,input_seq_len, target_pos, answers = batch
                    rec_output = self.seq_model(input_ids)
                    rec_output = rec_output[:,-1,:]

                    rating_pred = self.predict_full(rec_output)
                    if getattr(self.args, "eval_topk_on_gpu", True):
                        rating_pred = rating_pred.masked_fill(self.train_matrix_mask(user_ids), 0.0)
                        batch_pred_list = torch.topk(rating_pred, k=20, dim=1).indices.detach().cpu().numpy()
                    else:
                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        batch_user_index = user_ids.cpu().numpy()
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                        ind = np.argpartition(rating_pred, -20)[:, -20:]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    pred_batches.append(batch_pred_list)
                    answer_batches.append(answers.detach().cpu().numpy())

            pred_list = np.concatenate(pred_batches, axis=0)
            answer_list = np.concatenate(answer_batches, axis=0)

            metrices_list, log_info = self.get_full_sort_score(epoch, answer_list, pred_list)


            return metrices_list, log_info
                    
    
    def calculate_diff_loss(self, model, x_start, s, t, noise=None, loss_type="l2"):

        if noise is None:
            noise = torch.randn_like(x_start) 
        
        x_noisy = model.forward_process(x_start=x_start, t=t, noise=noise)

        predicted_x = model(x_noisy, s, t)
        
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x 
    
    def calculate_cl_loss(self, x0_hat_rep, x0_rep, target):
        batch_size = x0_rep.shape[0]
        sem_nce_logits, sem_nce_labels = self.info_nce(x0_hat_rep,x0_rep,self.args.temperature, batch_size, self.sim, target)
        cl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cl_loss

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot',intent_id=None):

        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self.mask_correlated_samples(intent_id)
        negative_samples = sim
        negative_samples[mask==0]=float("-inf")

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels
    
    def mask_correlated_samples(self, label):
        label=label.view(1,-1)
        label=label.expand((2,label.shape[-1])).reshape(1,-1)
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())
        return mask==0

