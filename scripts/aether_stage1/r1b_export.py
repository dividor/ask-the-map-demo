"""R1b export, run in AETHER's own environment: reload their trained SatBird model (our reconstruction
configs),
run their test loop, and save what the ask-the-map port needs to recompute the same numbers.

Usage: python r1b_export.py <aether_repo> <reconstruction_configs_dir> <checkpoint> <seed> <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    repo, rec_dir, ckpt, seed, out_dir = sys.argv[1:6]
    sys.path.insert(0, repo)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import hydra
    from hydra import compose, initialize_config_dir
    from lightning import Trainer, seed_everything
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("str", str, replace=True)
    with initialize_config_dir(config_dir=f"{repo}/configs", version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=satbird_alignment/aef_s01",
                f"seed={seed}",
                f"hydra.searchpath=[file://{rec_dir}]",
                "data.num_workers=0",
                "data.persistent_workers=false",
                "~data.prefetch_factor",
            ],
            return_hydra_config=True,
        )
    seed_everything(int(seed), workers=True)
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    trainer = Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    results = trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt, weights_only=False)[0]

    model.eval()
    geo, aux = [], []
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            g = model.geo_encoder(batch)
            g = model.geo_adapter(g)
            geo.append(g.float())
            aux.append(batch["aux"]["aux"].float())
        geo = torch.cat(geo)
        aux = torch.cat(aux)
        concepts = model.concepts
        cemb = F.normalize(model.text_encoder({"text": concepts}, mode="train"), dim=1).float()
        sim = model.concept_similarities(geo).float()

    # one training batch for the loss check
    seed_everything(int(seed), workers=True)
    batch = next(iter(datamodule.train_dataloader()))
    with torch.no_grad():
        g, t = model.forward(batch, "train")
        loss = model.loss_fn(
            g, t, mode="train", aux_values=batch["aux"]["aux_std"], aux_ids_per_caption=batch["text_aux_ids"]
        )

    np.savez_compressed(
        out / f"seed_{seed}.npz",
        geo=geo.numpy(),
        aux=aux.numpy(),
        concept_embeds=cemb.numpy(),
        similarity=sim.numpy(),
        loss_geo=g.float().numpy(),
        loss_text=t.float().numpy(),
        loss_aux_std=batch["aux"]["aux_std"].float().numpy(),
        log_temp=model.loss_fn.log_temp.detach().float().numpy(),
    )
    meta = {
        "seed": int(seed),
        "checkpoint": ckpt,
        "their_test_metrics": {k: float(v) for k, v in results.items()},
        "concept_configs": [
            {k: (v if not isinstance(v, np.floating | np.integer) else float(v)) for k, v in c.items()}
            for c in model.concept_configs
        ],
        "concept_names": list(model.concept_names),
        "baselines_test": {k: float(v) for k, v in model.dynamic_k_baselines["test"].items()},
        "loss_ids": [list(map(int, x)) for x in batch["text_aux_ids"]],
        "sigma": float(model.loss_fn.sigma),
        "their_loss": float(loss),
    }
    (out / f"seed_{seed}.json").write_text(json.dumps(meta, indent=1))
    print("saved", out / f"seed_{seed}.npz", "test metrics:", len(results))


if __name__ == "__main__":
    main()
