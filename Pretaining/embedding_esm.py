import esm
import torch
from torch.nn import Identity
import pickle
import pathlib

from embedding import Embedding


class EsmEmbedding(Embedding):
    def __init__(self, pooling="mean"):
        # model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        # model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.model = model
        self.alphabet = alphabet
        self.model.contact_head = Identity()
        self.model.emb_layer_norm_after = Identity()
        self.model.lm_head = Identity()
        self.model.eval()
        self.batch_converter = alphabet.get_batch_converter()
        self.embed_dim = model.embed_dim
        self.pooling = pooling
        # Freeze the parameters of ESM
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_embedding(self, batch):
        data = [
            (target, seq) for target, seq in zip(batch["target"], batch["sequence"])
        ]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            esm_result = self.model(batch_tokens)

        # The first token of every sequence is always a special classification token ([CLS]).
        # The final hidden state corresponding to this token is used as the aggregate sequence representation
        # for classification tasks.
        return self._pooling(self.pooling, esm_result["logits"], batch_tokens, self.alphabet.padding_idx)

    def to(self, device):
        self.model = self.model.to(device)

    def get_embedding_dim(self):
        return self.model.embed_dim

    def _pooling(self, strategy, tensors, batch_tokens, pad_token_id):
        """Perform pooling on [batch_size, seq_len, emb_dim] tensor

        Args:
            strategy: One of the values ["mean", "max", "cls"]
        """
        if strategy == "cls":
            seq_repr = tensors[:, 0, :]
        elif strategy == "mean":
            seq_repr = []
            batch_lens = (batch_tokens != pad_token_id).sum(1)

            for i, tokens_len in enumerate(batch_lens):
                seq_repr.append(tensors[i, 1 : tokens_len - 1].mean(0))

            seq_repr = torch.vstack(seq_repr)
        else:
            raise NotImplementedError("This type of pooling is not supported")
        return seq_repr

    def store_embeddings(self, batch, out_dir):
        """Store each protein embedding in a separate file named [protein_id].pkl

        Save all types of poolings such that each file has a [3, emb_dim]
        where rows 0, 1, 2 are mean, max, cls pooled respectively

        Args:
            batch: Each sample contains protein_id and sequence
        """
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        data = [
            (protein_id, seq)
            for protein_id, seq in zip(batch["protein_id"], batch["sequence"])
        ]
        batch_labels, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            esm_result = self.model(batch_tokens)

        esm_result = esm_result["logits"].detach().cpu()
        mean_max_cls_embeddings = []
        mean_embeddings = self._pooling("mean", esm_result, batch_tokens)
        cls_embeddings = self._pooling("cls", esm_result, batch_tokens)

        for protein_id, mean_emb, cls_emb in zip(
            batch_labels, mean_embeddings, cls_embeddings
        ):
            embeddings_dict = {
                "mean": mean_emb.numpy(),
                "cls": cls_emb.numpy(),
            }
            with open(f"{out_dir}/{protein_id}.pkl", "wb") as file:
                pickle.dump(embeddings_dict, file)