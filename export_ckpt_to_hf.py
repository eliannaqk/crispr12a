# export_ckpt_to_hf.py
#
# Convert a Composer / Lightning .ckpt into a Hugging-Face folder.
# Works with mosaicml-composer ≥ 0.30 (no need for the old hf_export CLI).

from composer.utils.export_pytorch_model import export_with_hf_format
from pathlib import Path

ckpt   = Path("/gpfs/gibbs/pi/gerstein/crispr/models/zenodo_15128064/"
              "progen2-base-ft-crispr.ckpt")
cfg_js = Path("/gpfs/gibbs/pi/gerstein/crispr/models/zenodo_15128064/"
              "progen2-base-ft-crispr-config.json")
outdir = Path("/gpfs/gibbs/pi/gerstein/crispr/models/zenodo_15128064/"
              "huggingface")

outdir.mkdir(parents=True, exist_ok=True)

export_with_hf_format(
    checkpoint_path   = ckpt,
    output_path       = outdir,
    hf_config_path    = cfg_js,        # copied to outdir/config.json
    shard_size_bytes  = 2_000_000_000  # 2-GB *.bin shards
)

print("✓ Export finished — HF files written to", outdir)