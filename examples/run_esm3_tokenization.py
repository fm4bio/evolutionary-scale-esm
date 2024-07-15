# %%
import sys

sys.path.append("../")

import os
import glob
import argparse
import pandas as pd
import time
from tqdm import tqdm

import torch
import torch.distributed as dist

from esm.models.esm3 import ESM3
from esm.utils import encoding, decoding
from esm.sdk.api import ESMProtein

import biotools.protein as protein
from biotools.protein import to_sequence, apply_residue_mask, residue_constants
from biotools.kabsch.kabsch import find_rigid_alignment


PDB_ROOT = "../../datasets/pdb/structures"
AFDB_ROOT = "../../../datasets/afdb/proteomes"


def load_model(device="cpu"):
    model = ESM3.from_pretrained("esm3_sm_open_v1", device=device).eval()
    structure_token_encoder = model.get_structure_token_encoder()
    structure_tokenizer = model.tokenizers.structure
    structure_decoder = model.get_structure_token_decoder()
    del model
    return (structure_token_encoder, structure_tokenizer, structure_decoder)


def load_pdb_inner(pdb_file, chain_id=None, only_protein=False, afdb_plddt_thresh=70.0):
    if ":" in pdb_file:
        cif, tar, seek = pdb_file.split(":")
        tar = f"{AFDB_ROOT}/proteome-tax_id-{tar}_v4.tar"
        seek = int(seek)
        po = protein.from_afdb_tar(tar, seek, chain_id=chain_id)
        # drop low plddt region
        if afdb_plddt_thresh > 0.0:
            plddt_mask = po.b_factors[:, 1] >= afdb_plddt_thresh  # 1 is CA
            po = protein.apply_residue_mask(po, plddt_mask)
        return po  # AFDB only contains protein. no need to filter

    if pdb_file.endswith(".gz"):
        po = protein.from_pdb_gz(pdb_file, chain_id=chain_id)
    else:
        po = protein.from_pdb_file(pdb_file, chain_id=chain_id)
    if only_protein:
        is_protein = protein.residue_constants.is_aatype_protein(po.aatype)  # [num_res,]
        po = protein.apply_residue_mask(po, is_protein)
    return po


def load_pdb(pdb_file, chain_id=None):
    po = load_pdb_inner(pdb_file, chain_id=chain_id, only_protein=True, afdb_plddt_thresh=0.0)
    po.atom_positions[po.atom_mask == 0] = float("nan")
    coords = torch.Tensor(po.atom_positions[:, :37, :])
    seq = to_sequence(po, only_protein=True)
    return ESMProtein(sequence=seq, coordinates=coords)


def get_pdb_path_chain_id_afdb(df, idx):
    row = df.iloc[idx]
    pdb_path = f"{row['cif']}:{row['tar']}:{row['seek']}"
    chain_id = None  # if None, it returns all chains
    return pdb_path, chain_id


def get_pdb_path_chain_id(df, idx):
    row = df.iloc[idx]
    if "pdb_path" in row:
        pdb_path = row["pdb_path"]
    else:  # support pdb_id
        pdb_id = row["pdb_id"]
        pdb_path = f"{PDB_ROOT}/{pdb_id[1:3]}/pdb{pdb_id}.ent.gz"
    if "chain_id" in row:
        chain_id = row["chain_id"]
    else:
        chain_id = None  # if None, it returns all chains
    return pdb_path, chain_id


class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv):
        self.csv = csv
        self.df = pd.read_csv(csv)

    def __getitem__(self, idx):
        if "afdb" in self.csv:
            pdb_path, chain_id = get_pdb_path_chain_id_afdb(self.df, idx)
        else:
            pdb_path, chain_id = get_pdb_path_chain_id(self.df, idx)
        input_protein = load_pdb(pdb_path, chain_id)
        return (idx, input_protein)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def collate_fn(batch):
        return list(zip(*batch))


def tokenize(coordinates, structure_token_encoder, structure_tokenizer):
    _coordinates, _plddt, structure_tokens = encoding.tokenize_structure(
        coordinates,  # [BOS] and [EOS] will be automatically added
        structure_encoder=structure_token_encoder,
        structure_tokenizer=structure_tokenizer,
        add_special_tokens=True,
    )
    return structure_tokens


def decode(structure_tokens, structure_decoder, structure_tokenizer):
    # only decodes backbone atoms
    coordinates, plddt, ptm = decoding.decode_structure(
        structure_tokens,  # [BOS] and [EOS] will be automatically ignored
        structure_decoder,
        structure_tokenizer,
    )
    return ESMProtein(coordinates=coordinates, plddt=plddt, ptm=ptm)


def post_processing(decoded_protein, input_protein):
    # set the sequence to the input sequence
    decoded_protein.sequence = input_protein.sequence
    # align the decoded coordinates to the input coordinates
    ca1, ca2 = decoded_protein.coordinates[:, 1, :], input_protein.coordinates[:, 1, :]
    # mask of inf or nan
    inf_or_nan = torch.isnan(ca1) | torch.isnan(ca2) | torch.isinf(ca1) | torch.isinf(ca2)
    inf_or_nan = inf_or_nan.any(dim=-1)
    R, t = find_rigid_alignment(ca1, ca2, mask=(~inf_or_nan).float())
    # apply alignment
    decoded_protein.coordinates = decoded_protein.coordinates @ R.T + t
    return decoded_protein


def save_pdb(protein, fname):
    return protein.to_pdb(fname)


def task(args, model_pack, input_protein):
    structure_token_encoder, structure_tokenizer, structure_decoder = model_pack
    structure_tokens = tokenize(input_protein.coordinates, structure_token_encoder, structure_tokenizer)
    if args.decode:
        decoded_protein = decode(structure_tokens, structure_decoder, structure_tokenizer)
        decoded_protein = post_processing(decoded_protein, input_protein)
    else:
        decoded_protein = None
    return {
        "input": input_protein,
        "tokens": structure_tokens,
        "decoded": decoded_protein,
    }


def init(args):
    global WORLD_RANK, WORLD_SIZE, device

    if args.distributed:
        dist.init_process_group("nccl")
        WORLD_RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        device = int(os.environ.get("LOCAL_RANK", "0"))

        time.sleep(device * 0.1)
        print(f"hi, I am the {WORLD_RANK} of {WORLD_SIZE} and my device is {device}", flush=True)
        device = torch.device(f"cuda:{device}")
    else:
        WORLD_RANK = 0
        WORLD_SIZE = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return WORLD_RANK, WORLD_SIZE, device


def destroy(args):
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()

    WORLD_RANK, WORLD_SIZE, device = init(args)
    model_pack = load_model(device)
    torch.set_grad_enabled(False)

    if "*" in args.csv:
        csvs = sorted(glob.glob(args.csv))
    else:
        csvs = args.csv.split(",")
    for csv in csvs:
        output_dir = f"{os.path.basename(csv)}"

        # make output directory
        if WORLD_RANK == 0:
            os.makedirs(f"{output_dir}", exist_ok=True)
            if args.decode:
                os.makedirs(f"{output_dir}/pdb", exist_ok=True)
        if args.distributed:
            dist.barrier()

        ds = Dataset(csv)
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=WORLD_SIZE, rank=WORLD_RANK, shuffle=False, drop_last=False)
        else:
            sampler = None
        dataloader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=32, collate_fn=Dataset.collate_fn, sampler=sampler)
        pbar = tqdm(total=len(ds) // WORLD_SIZE)
        with open(f"{output_dir}/{WORLD_RANK}.txt", "w") as fo:
            for idx, data in dataloader:
                results = []
                for idx_i, data_i in zip(idx, data):
                    result_i = task(args, model_pack, data_i)
                    results.append(result_i)
                # save the result
                for idx_i, result_i in zip(idx, results):
                    row_i = ds.df.iloc[idx_i]
                    if "afdb" in csv:
                        pdb_path = f"{row_i['cif']}:{row_i['tar']}:{row_i['seek']}"
                        chain_id = None
                        fo.write(f">{pdb_path}\n")
                    else:
                        pdb_path = row_i["pdb_path"] if "pdb_path" in row_i else row_i["pdb_id"]
                        chain_id = row_i["chain_id"] if "chain_id" in row_i else None
                        fo.write(f">{pdb_path}_{chain_id}\n")
                    fo.write(" ".join([str(x) for x in result_i["tokens"].tolist()]) + "\n")
                    if args.decode:
                        save_pdb(result_i["input"], f"{output_dir}/pdb/{idx_i}_target.pdb")
                        save_pdb(result_i["decoded"], f"{output_dir}/pdb/{idx_i}_dec.pdb")
                pbar.update(len(idx))
        pbar.close()


# example usage:

# OMP_NUM_THREADS=12 torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id 20240115 --rdzv_backend c10d --rdzv_endpoint gpumid-49:29500 \
# run_esm3_tokenization.py --distributed --csv ~/hom/protein-x/esm/equiformer/data/notes/profile_resol4A_20220501_train_with_sampler_clean.csv

# OMP_NUM_THREADS=12 torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id 20240115 --rdzv_backend c10d --rdzv_endpoint gpumid-49:29500 \
#     run_esm3_tokenization.py --distributed --csv $HOME"/hom/datasets/afdb/split_cluster/train_csv/afdb_40_*_100.csv"

# OMP_NUM_THREADS=12 torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id 20240115 --rdzv_backend c10d --rdzv_endpoint gpumid-49:29500 \
#     run_esm3_tokenization.py --distributed --csv xinyi/xinyi_pdb.csv
