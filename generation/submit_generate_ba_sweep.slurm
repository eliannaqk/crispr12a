#!/bin/bash -l
#SBATCH --job-name=gen-ba-sweep
#SBATCH -A pi_mg269
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o slurm-generate-sweep-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Environment: oc-opencrispr per project policy
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate oc-opencrispr
echo "[Env] Using conda env: oc-opencrispr"
# Mitigate CUDA fragmentation-related OOMs (PyTorch recommendation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/crispr12/opencrispr-repro-main
echo "[CWD] $(pwd)"

# Model path to sweep. Override via environment if desired.
# Examples:
#  export MODEL_PATH=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-111648/huggingface/ba8000/
#  export MODEL_PATH=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-202947/huggingface/ba8000/
MODEL_PATH=${MODEL_PATH:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250828-193254/huggingface/ba4000/}

# Output root. All combos write separate files by config name under this folder.
GEN_ROOT=${GEN_ROOT:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences}
mkdir -p "$GEN_ROOT"

# Derive run/model tags for save folder
MP_NOSLASH=${MODEL_PATH%/}
MODEL_TAG=${MP_NOSLASH##*/}
RUN_TAG=$(echo "$MODEL_PATH" | sed -E 's|.*/(run-[^/]+)/huggingface/.*|\1|')
SWEEP_TAG=sweep
SAVE_FOLDER="$GEN_ROOT/${RUN_TAG}-${MODEL_TAG}-${SWEEP_TAG}"
mkdir -p "$SAVE_FOLDER"

echo "[Run] Sweep for MODEL=${MODEL_PATH} -> SAVE_FOLDER=${SAVE_FOLDER}"

# Context sequences
# SEED1 and SEED3 remain explicit. SEED2 is derived from seq1.csv so it exactly matches the Lb anchor tail (Lb positions 514..1228).
SEED1_SEQ='1MSKLEKFTNCYSLSKTLRFKAIPVGKTQENIDNKRLLVEDEKRAEDYKGVKKLLDRYYLSFINDVLHSIKLKNLNNYISLFRKKTRTEKENKELENLEINLRKEIAKAFKGNEGYKSLFKKDIIETILPEFLDDKDEIALVNSFNGFTTAFTGFFRNRENMFSEEAKSTSIAFRCINENLTRYISNMDIFEKVDAIFDKHEVQEIKEKILNSDYDVEDFFEGEFFNFVLTQEGIRVYNAIIGGFVTESGEKIKGLNEYINLYNQKTKQKLPKFKPLYKQVLSDRESLSFYGRGYTSDEEVLEVFRNTLNKNSEIFSSIKKLEKLFKNFDEYSSAGIFVKNGPAISTISKRIFGEWNVIRDKWNAEYDDIHLKKKAVVTEKYEDDRRKSFKKIGSFSLEQLQEYADADLSVVEKLKEIIIQKVDEIYKVYGSSEKLFDADFVLEKSLKKNDAVVAIMKDLLDSVKSFENYIKAFFGEGKETNRDESFYGDFVLAYDILLKVDHIYDAIRNYVTQ'
SEED3_SEQ='1MSKLEKFTNCYSLSKTLRFKAIPV'

# Compute SEED2 from seq1.csv (exact Lb tail positions 514..1228), then REVERSE it
SEED2_PROT=$(python - << 'PY'
import csv
with open('seq1.csv', newline='') as f:
    r=csv.DictReader(f)
    rows=list(r)
    if not rows or 'sequence' not in r.fieldnames:
        raise SystemExit(1)
    anc=''.join(ch for ch in rows[0]['sequence'].strip().upper() if ch.isalpha())
start,end=514,1228
sub=anc[start-1:end]
print(sub[::-1])
PY
)
SEED2_SEQ="2${SEED2_PROT}"

# Original SEED4 remains explicit
SEED4_SEQ='2HKVSTQAYELWEKNSIAIKVKDLKEDEAKKFQGIAWLVKRAINYAGNADANKPLIANEQAEYNRSDYFIGDSNKVPSILFDVDTRGTISNRMQLMLSMLAMFSSYFAKDSQECLLARIDGQQYNIGYKNFLEKYASTLCVEEWDFVNNKKPNRFIRIRNGYSYLKWKKIYDADTRSFNKYDLAFEFLDEEPVYMIRDFSSIFKKSDAISTYKTKLLNVFGTSPDIKSTLWAPIYFIFGNQTSMSKFSEFKNTIQYGKLAGGTACPNSKKDVMYNLKDILMKEFKQYVQKEVKVRSNKFGSNLDELAIVADYKEVLECIKHVVQSIYGAKLEKINEISTWNQRAEFREKEKKDLLSHYDTKIRIGNFNNIIENLSYQEVINGKGDVVVIYLLNREGRAIGIVYPNDDHKLLVRVETNIKFINKPCKNIAIPIHLEYQDESFRKDKYVDYSLTTTKKPNDPNKNAIPSNAPHVVLEEKKLSARRMFLEAGGSLRIQGHNNEDFLLKFYMTHLNPTGHSKDSFDKNYIQFMYLKGEEVLKDVEKKSASEFSVKYGQEEVERYFGAIDKYKETESFNFDYANSWKPYRSISDKFFDILKHCDNLNFMDGKKFTGNKYIKQIDESPNYYAMWKKSFFVKPLMKNPGPLLKYNIKEYNGNVDDKDIKQLCKAYKKDMIALYYKSGYRLITARYDTEKDKDWGGMFQPNQFYLKFKDKSYPKQTVYNRIADYIHDVKLLIDYALVFDGYFSEDRNTEKGEGFFAKIYNEFSKVSDLLDKMIAVVADNKKLSKELVFDADFLKESSGYVKYIEDVKQIIIEKLKEVVSLDADAYEQLQELSFSGIKKFSKRRDDEYKETVVAKKKLHIDDYEANWKDRIVNWEGFIRKSITSIAPGNKVFIGASSYEDFNKFLKELKKISSFIESNKNLTNRFVELVEEDSTYGRGYFSLSERDSLVQKYLPKFKPLKQKTKQNYLNIYENLGKIKEGSETVFGGIIANYVRIGEQTLVFNFFEGEFFDEVDYDSNLIKEKIEQVEHKDFIADVKEFIDMNSIYRTLNENICRFAISTSKAEESFMNERNRFFGTFATTFGNFSNVLAIEDKDDLFEPLITEIIDKKFLSKYGENGKFAKAIEKRLNIELNELEKNEKETRTKKRFLSIYNNLNKLKISHLVDNIFSLYYRDLLKKVGKYDEARKEDEVLLRKNDINEQTKG'

# Map to sweep sets
SEQ1="${SEED1_SEQ}"
SEQ2="${SEED2_SEQ}"
SEQ3="${SEED3_SEQ}"
SEQ4="${SEED4_SEQ}"

# Six sampling combos to sweep (T, top_p)
# We include both top_p=0.5 and 0.95 at T in {0.5, 0.7, 1.0}
declare -a TEMPS=(0.5 0.5 0.7 0.7 1.0 1.0)
declare -a TOPPS=(0.5 0.95 0.5 0.95 0.5 0.95)

# Build per-combo YAMLs with exactly one combo per context so each run yields 10k per context
CFG_DIR=${CFG_DIR:-generation/sweep_configs}
mkdir -p "$CFG_DIR"

write_cfgs() {
  local tag="$1"; shift
  local seq_a="$1"; shift
  local seq_b="$1"; shift
  local nsamp="$1"; shift
  for i in ${!TEMPS[@]}; do
    T=${TEMPS[$i]}
    P=${TOPPS[$i]}
    TP_TAG="${tag}_t${T}_p${P}"
    CFG_PATH="$CFG_DIR/generate_50k_txp_${TP_TAG}.yml"
    {
      if [[ -n "$seq_a" ]]; then
        cat << EOF
- context:
    name: "crispr12a"
    seq: "${seq_a}"

  num_samples: ${nsamp}
  batch_size: 64
  do_sample: true
  temperature: ${T}
  top_p: ${P}
  top_k: 0
  repetition_penalty: 1.02
EOF
      fi
      if [[ -n "$seq_b" ]]; then
        cat << EOF

- context:
    name: "crispr12a"
    seq: "${seq_b}"

  num_samples: ${nsamp}
  batch_size: 64
  do_sample: true
  temperature: ${T}
  top_p: ${P}
  top_k: 0
  repetition_penalty: 1.02
EOF
      fi
    } > "$CFG_PATH"
    echo "[cfg] Wrote $CFG_PATH (nsamp=${nsamp})"
  done
}

# Build per-seed configs with their own contexts, 20k per combo
write_cfgs seed1 "${SEED1_SEQ}" "" 20000
write_cfgs seed2 "${SEED2_SEQ}" "" 20000
write_cfgs seed3 "${SEED3_SEQ}" "" 20000
write_cfgs seed4 "${SEED4_SEQ}" "" 20000

SEED_SET=${SEED_SET:-}
if [[ -n "$SEED_SET" ]]; then
  echo "[Sweep] Restricted to SEED_SET=$SEED_SET"
  TAGS=("$SEED_SET")
else
  echo "[Sweep] Launching ${#TEMPS[@]} generate.py runs for: seed1, seed2, seed3, seed4"
  TAGS=(seed1 seed2 seed3 seed4)
fi
for tag in "${TAGS[@]}"; do
  for i in ${!TEMPS[@]}; do
    T=${TEMPS[$i]}
    P=${TOPPS[$i]}
    TP_TAG="${tag}_t${T}_p${P}"
    CFG_PATH="$CFG_DIR/generate_50k_txp_${TP_TAG}.yml"
    if [[ -s "$CFG_PATH" ]]; then
      echo "[eval] ${tag} combo t=${T} p=${P} -> $CFG_PATH"
      # Continue to next combo on failure (e.g., OOM) so all temps submit
      python -u generate.py \
        --model-path "$MODEL_PATH" \
        --config "$CFG_PATH" \
        --save-folder "$SAVE_FOLDER" || { echo "[warn] generation failed for ${tag} T=${T} P=${P}"; continue; }
    else
      echo "[skip] Empty config for ${tag} t=${T} p=${P} (no contexts)"
    fi
  done
done

echo "[Done] Sweep complete. Outputs under: $SAVE_FOLDER/generations/${MODEL_TAG}/"
