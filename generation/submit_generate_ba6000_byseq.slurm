#!/bin/bash -l
#SBATCH --job-name=gen-ba6000-byseq
#SBATCH -A pi_mg269
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o slurm-generate-ba6000-byseq-%j.out

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ENV: approved env for generation
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate oc-opencrispr
echo "[Env] Using conda env: oc-opencrispr"

cd ~/crispr12/opencrispr-repro-main

MODEL_PATH="${MODEL_PATH:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-202947/huggingface/ba6000}"
GEN_ROOT=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences
mkdir -p "$GEN_ROOT"
MP_NOSLASH=${MODEL_PATH%/}
MODEL_TAG=${MP_NOSLASH##*/}
RUN_TAG=$(echo "$MODEL_PATH" | sed -E 's|.*/(run-[^/]+)/huggingface/.*|\1|')
SAVE_BASE="$GEN_ROOT/${RUN_TAG}-${MODEL_TAG}-byseq"
BASE_CFG="generation/generate_50k_txp.yml"  # 6x (T,P) combos; context.seq will be overwritten per seed

mkdir -p "$SAVE_BASE"

# Three input seed sequences (1..., 2..., 1...)
SEED1="1MSKLEKFTNCYSLSKTLRFKAIPVGKTQENIDNKRLLVEDEKRAEDYKGVKKLLDRYYLSFINDVLHSIKLKNLNNYISLFRKKTRTEKENKELENLEINLRKEIAKAFKGNEGYKSLFKKDIIETILPEFLDDKDEIALVNSFNGFTTAFTGFFRNRENMFSEEAKSTSIAFRCINENLTRYISNMDIFEKVDAIFDKHEVQEIKEKILNSDYDVEDFFEGEFFNFVLTQEGIRVYNAIIGGFVTESGEKIKGLNEYINLYNQKTKQKLPKFKPLYKQVLSDRESLSFYGRGYTSDEEVLEVFRNTLNKNSEIFSSIKKLEKLFKNFDEYSSAGIFVKNGPAISTISKRIFGEWNVIRDKWNAEYDDIHLKKKAVVTEKYEDDRRKSFKKIGSFSLEQLQEYADADLSVVEKLKEIIIQKVDEIYKVYGSSEKLFDADFVLEKSLKKNDAVVAIMKDLLDSVKSFENYIKAFFGEGKETNRDESFYGDFVLAYDILLKVDHIYDAIRNYVTQ"
SEED2="2HKVSTQAYELWEKNSIAIKVKDLKEDEAKKFQGIAWLVKRAINYAGNADANKPLIANEQAEYNRSDYFIGDSNKVPSILFDVDTRGTISNRMQLMLSMLAMFSSYFAKDSQECLLARIDGQQYNIGYKNFLEKYASTLCVEEWDFVNNKKPNRFIRIRNGYSYLKWKKIYDADTRSFNKYDLAFEFLDEEPVYMIRDFSSIFKKSDAISTYKTKLLNVFGTSPDIKSTLWAPIYFIFGNQTSMSKFSEFKNTIQYGKLAGGTACPNSKKDVMYNLKDILMKEFKQYVQKEVKVRSNKFGSNLDELAIVADYKEVLECIKHVVQSIYGAKLEKINEISTWNQRAEFREKEKKDLLSHYDTKIRIGNFNNIIENLSYQEVINGKGDVVVIYLLNREGRAIGIVYPNDDHKLLVRVETNIKFINKPCKNIAIPIHLEYQDESFRKDKYVDYSLTTTKKPNDPNKNAIPSNAPHVVLEEKKLSARRMFLEAGGSLRIQGHNNEDFLLKFYMTHLNPTGHSKDSFDKNYIQFMYLKGEEVLKDVEKKSASEFSVKYGQEEVERYFGAIDKYKETESFNFDYANSWKPYRSISDKFFDILKHCDNLNFMDGKKFTGNKYIKQIDESPNYYAMWKKSFFVKPLMKNPGPLLKYNIKEYNGNVDDKDIKQLCKAYKKDMIALYYKSGYRLITARYDTEKDKDWGGMFQPNQFYLKFKDKSYPKQTVYNRIADYIHDVKLLIDYALVFDGYFSEDRNTEKGEGFFAKIYNEFSKVSDLLDKMIAVVADNKKLSKELVFDADFLKESSGYVKYIEDVKQIIIEKLKEVVSLDADAYEQLQELSFSGIKKFSKRRDDEYKETVVAKKKLHIDDYEANWKDRIVNWEGFIRKSITSIAPGNKVFIGASSYEDFNKFLKELKKISSFIESNKNLTNRFVELVEEDSTYGRGYFSLSERDSLVQKYLPKFKPLKQKTKQNYLNIYENLGKIKEGSETVFGGIIANYVRIGEQTLVFNFFEGEFFDEVDYDSNLIKEKIEQVEHKDFIADVKEFIDMNSIYRTLNENICRFAISTSKAEESFMNERNRFFGTFATTFGNFSNVLAIEDKDDLFEPLITEIIDKKFLSKYGENGKFAKAIEKRLNIELNELEKNEKETRTKKRFLSIYNNLNKLKISHLVDNIFSLYYRDLLKKVGKYDEARKEDEVLLRKNDINEQTKG"
SEED3="1MSKLEKFTNCYSLSKTLRFKAIPV"

combine_and_run() {
  local seed1="$1"; local seed2="$2"; local seed3="$3"
  local ctx="${seed1}${seed2}${seed3}"
  # Make a slug based on lengths and heads, plus short hash
  python - "$ctx" <<'PY'
import sys, re, hashlib, json
ctx = sys.argv[1]
def mk_slug(ctx):
    parts = re.split(r'([12])', ctx)
    seqs = []
    for i in range(1, len(parts), 2):
        if i+1 < len(parts):
            t, s = parts[i], parts[i+1]
            if t in {'1','2'} and s:
                seqs.append((t, s))
    items=[]
    for t,s in seqs[:3]:
        items.append(f"{t}{len(s)}_{re.sub('[^A-Z]','',s)[:6]}")
    h = hashlib.sha1(ctx.encode()).hexdigest()[:8]
    return f"seed_{'_'.join(items)}_{h}"
slug = mk_slug(ctx)
print(slug)
PY
}

run_for_seed() {
  local ctx_full="$1"
  local slug="$2"
  local out_cfg="/tmp/generate_50k_txp_${slug}.yml"
  # Overwrite context.seq in every entry of BASE_CFG and write to out_cfg
  python - "$BASE_CFG" "$ctx_full" "$out_cfg" <<'PY'
import sys
from ruamel.yaml import YAML
inp, ctx, out = sys.argv[1], sys.argv[2], sys.argv[3]
yaml = YAML(typ='safe')
data = yaml.load(open(inp))
if isinstance(data, dict):
    data = [data]
for item in data:
    if 'context' in item and isinstance(item['context'], dict):
        item['context']['seq'] = ctx
yaml = YAML()
yaml.default_flow_style = False
with open(out, 'w') as f:
    yaml.dump(data, f)
print(out)
PY
  local save_dir="$SAVE_BASE/$slug"
  mkdir -p "$save_dir"
  echo "[Run] Seed $slug -> $save_dir"
  python -u generate.py \
    --model-path "$MODEL_PATH" \
    --config "$out_cfg" \
    --save-folder "$save_dir"
}

slug=$(combine_and_run "$SEED1" "$SEED2" "$SEED3")
ctx_full="${SEED1}${SEED2}${SEED3}"
run_for_seed "$ctx_full" "$slug"

echo "[Done] Check outputs under: $SAVE_BASE/$slug/generations/ba6000/"
