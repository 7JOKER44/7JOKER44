#!/usr/bin/env python3
# Disable HF symlink warnings & widget progress bars
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
from datasets import logging as ds_logging
try:
    ds_logging.disable_progress_bar()
except AttributeError:
    pass

# federated_rag_healthcare_privacy_comparison_quicktest.py

# === Dependencies ===
# Assumes required packages are pre-installed in your environment:
# tenseal, secretsharing, sentencepiece, hf_xet, fsspec, flwr, datasets, transformers, torch, opacus

# === 1. Monkey-patch for secretsharing lib ===
import builtins
setattr(builtins, "long", int)

# === 2. Standard imports ===
import random
import numpy as np
import tenseal as ts
from secretsharing import SecretSharer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import flwr as fl
from opacus import PrivacyEngine
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt

# === Quick-test constants ===
CLIENT_NUM = 2           # Simulated clients for fast testing
NUM_ROUNDS = 1         # Single federated round
SMALL_DATASET_SIZE = 100  # Cap total samples for speed

# === Model identifiers ===
RETRIEVER_MODEL = "sentence-transformers/all-mpnet-base-v2"
GENERATOR_MODEL = "t5-small"

# === Global store for secret-shares ===
SHARES = {}

# === Load & split PubMedQA (small subset) ===
def load_and_split(seed: int = 42):
    ds = load_dataset("pubmed_qa", "pqa_labeled")
    data = ds["train"]
    # Restrict to small subset if set
    if SMALL_DATASET_SIZE and SMALL_DATASET_SIZE < len(data):
        data = data.select(range(SMALL_DATASET_SIZE))
    data = data.shuffle(seed=seed)
    chunk = len(data) // CLIENT_NUM
    client_splits = [data[i*chunk:(i+1)*chunk] for i in range(CLIENT_NUM)]
    test_size = min(chunk, 50)
    test_set = data[-test_size:]
    return client_splits, test_set

# === Dataset wrapper ===
class PubMedQADataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.gen_tokenizer = T5Tokenizer.from_pretrained(GENERATOR_MODEL)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        q, a = item["question"], item["answer"]
        gen_enc = self.gen_tokenizer(
            q, truncation=True, padding="max_length",
            max_length=128, return_tensors="pt"
        )
        labels = self.gen_tokenizer(
            a, truncation=True, padding="max_length",
            max_length=32, return_tensors="pt"
        )["input_ids"]
        return {"gen_enc": {k: v.squeeze(0) for k, v in gen_enc.items()}, "labels": labels}

# === Federated client class ===
class RAGClient(fl.client.NumPyClient):
    def __init__(self, cid, samples, use_dp=False, use_he=False, use_ss=False):
        self.cid = cid
        self.use_dp = use_dp
        self.use_he = use_he
        self.use_ss = use_ss
        self.dataset = PubMedQADataset(samples)

        # Generator only for quick test
        self.gen_model = T5ForConditionalGeneration.from_pretrained(GENERATOR_MODEL)
        self.gen_opt = AdamW(self.gen_model.parameters(), lr=5e-5)

        # DP parameters
        self.dp_noise, self.dp_max_grad = 1.0, 1.0

        # HE/SS setup
        if self.use_he or self.use_ss:
            ctx = ts.context(
                ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[40, 20, 40]
            )
            ctx.generate_galois_keys()
            ctx.global_scale = 2**40
            ser = ctx.serialize()
            if self.use_ss:
                shares = SecretSharer.split_secret(ser.hex(), CLIENT_NUM, CLIENT_NUM//2 + 1)
                SHARES[self.cid] = shares[int(self.cid)]
            else:
                self.secret_ctx = ctx
            self.public_ctx = ts.context_from(ser, inplace=False, load_secret_key=False)

    def get_parameters(self):
        state = self.gen_model.state_dict()
        params = []
        for v in state.values():
            arr = v.cpu().numpy()
            if self.use_he:
                flat = arr.flatten().tolist()
                ck = ts.ckks_vector(self.public_ctx, flat)
                params.append(ck.serialize())
            else:
                params.append(arr)
        return params

    def set_parameters(self, params):
        state = self.gen_model.state_dict()
        keys = list(state.keys())
        for idx, k in enumerate(keys):
            if self.use_he:
                if self.use_ss:
                    rec = SecretSharer.recover_secret([
                        SHARES[str(i)] for i in range(CLIENT_NUM//2 + 1)
                    ])
                    ctx = ts.context_from(bytes.fromhex(rec), inplace=False, load_secret_key=True)
                else:
                    ctx = self.secret_ctx
                ck = ts.ckks_vector_from(ctx, params[idx])
                dec = np.array(ck.decrypt(), dtype=np.float32)
                tensor = torch.from_numpy(dec.reshape(state[k].shape))
            else:
                tensor = torch.from_numpy(params[idx])
            state[k] = tensor
        self.gen_model.load_state_dict(state)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        if self.use_dp:
            engine = PrivacyEngine()
            self.gen_model, self.gen_opt, loader = engine.make_private(
                module=self.gen_model,
                optimizer=self.gen_opt,
                data_loader=loader,
                noise_multiplier=self.dp_noise,
                max_grad_norm=self.dp_max_grad,
            )
        self.gen_model.train()
        total_loss, count = 0.0, 0
        for batch in loader:
            ids = batch["gen_enc"]["input_ids"].to(self.gen_model.device)
            labels = batch["labels"].to(self.gen_model.device)
            out = self.gen_model(input_ids=ids, labels=labels)
            loss = out.loss
            self.gen_opt.zero_grad()
            loss.backward()
            self.gen_opt.step()
            total_loss += loss.item()
            count += 1
        return self.get_parameters(), len(self.dataset), {"loss": total_loss / count if count else 0.0}

    def evaluate(self, parameters, config):
        return 0.0, len(self.dataset), {"accuracy": 0.0}

# === Central evaluation ===
def evaluate_central(test):
    loader = DataLoader(PubMedQADataset(test), batch_size=4)
    gen = T5ForConditionalGeneration.from_pretrained(GENERATOR_MODEL).eval()
    correct, total = 0, 0
    for batch in loader:
        outs = gen.generate(input_ids=batch["gen_enc"]["input_ids"])
        for o, l in zip(outs, batch["labels"]):
            if gen.tokenizer.decode(o, skip_special_tokens=True) == gen.tokenizer.decode(l, skip_special_tokens=True):
                correct += 1
            total += 1
    return correct / total if total else 0.0

# === Main ===
def main():
    splits, test = load_and_split()
    methods = ["FL", "FL+DP", "FL+HE", "FL+SS", "FL+HE+SS"]
    histories, results = {}, {}
    for m in methods:
        dp = "DP" in m
        he = "HE" in m
        ss = "SS" in m
        print(f"Exp: {m}")
        clients = {
            str(i): RAGClient(str(i), splits[i], use_dp=dp, use_he=he, use_ss=ss)
            for i in range(CLIENT_NUM)
        }
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_eval=1.0,
            min_fit_clients=CLIENT_NUM,
            min_eval_clients=CLIENT_NUM,
        )
        history = fl.simulation.start_simulation(
            client_fn=lambda cid: clients[cid],
            num_clients=CLIENT_NUM,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            progress=False,
        )
        histories[m] = history
        acc = evaluate_central(test)
        results[m] = acc
        print(f"{m} accuracy: {acc:.4f}")
    plt.figure()
    for m in methods:
        losses = histories[m].metrics_distributed["fit"]["loss"]
        plt.plot(range(1, len(losses) + 1), losses, marker='o', label=m)
    plt.xlabel("Round")
    plt.ylabel("Avg. Loss")
    plt.title("Loss per Round")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()