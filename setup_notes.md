# Environment Setup Notes

Sufficient to reconstruct the Phase 1 environment on a fresh machine.

## Host

| Item | Value |
|------|-------|
| OS | Ubuntu 24.04.1 LTS (Noble Numbat) — running under WSL2 on Windows |
| WSL2 kernel | 6.6.87.2-microsoft-standard-WSL2 |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU (8 GiB VRAM) |
| CUDA driver | 566.36 (Windows host driver, exposed to WSL2) |
| CUDA toolkit | 12.4 (torch cu124 build; `nvcc` not required) |

## Software versions

| Item | Value |
|------|-------|
| Python | 3.12.3 (system, `/usr/bin/python3`) |
| Node.js | v20.20.2 (via nvm) |
| npm | (bundled with Node.js v20) |
| PyTorch | 2.6.0+cu124 |
| poke-env | 0.8.3.3 (UT-Austin-RPL fork, `recover_083` branch) |

## System packages installed (apt-get)

These were installed with `sudo apt-get install -y <package>` during Phase 1:

```
python3.12-dev      # Required for torch.compile / triton JIT (Python.h)
```

No other system packages were installed beyond the Ubuntu 24.04 defaults.

## Step-by-step reconstruction

### 1. Clone Pokemon Showdown (pinned commit)

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
git checkout 1d2258d0875e7b5f303f94ec0936a98535c37602
npm install
# Start server (keep running in background):
node pokemon-showdown start --no-security &
cd ..
```

Verify: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8000` should return `200`.

### 2. Create Python virtual environment

```bash
python3 -m venv metamon-env
source metamon-env/bin/activate
```

### 3. Install PyTorch with CUDA 12.4

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

CPU-only PyTorch is too slow for two concurrent transformer inference sessions.
GPU is required for D3 self-play to complete within a reasonable timeout.

### 4. Clone and install Metamon

```bash
git clone https://github.com/UT-Austin-RPL/metamon.git
cd metamon
git checkout main
# Apply the traj_save_len float bug fix (see DECISIONS.md):
# In metamon/rl/metamon_to_amago.py, the MetamonAMAGOExperiment(...) call must
# have traj_save_len=int(1e10) and stagger_traj_file_lengths=False.
pip install -e ".[all]"
pip uninstall -y pettingzoo   # conflicts with gymnasium version
cd ..
```

Set environment variables (add to shell profile or set before each run):

```bash
export METAMON_ALLOW_ANY_POKE_ENV=True    # poke-env 0.8.3.3 vs expected 0.8.3.2
export METAMON_CACHE_DIR=/path/to/metamon-cache
```

### 5. Install system headers for torch.compile

```bash
sudo apt-get install -y python3.12-dev
```

Without this, `VanillaAttention._inference_with_cache` (decorated with `@torch.compile`)
fails with `fatal error: Python.h: No such file or directory`.

### 6. Set METAMON_CACHE_DIR and pre-download checkpoint

```bash
export METAMON_CACHE_DIR=$HOME/metamon-cache
mkdir -p $METAMON_CACHE_DIR
source metamon-env/bin/activate
python - <<'EOF'
import os, sys
os.environ["METAMON_ALLOW_ANY_POKE_ENV"] = "True"
sys.path.insert(0, "metamon")
from metamon.rl.pretrained import SmallRLGen9Beta
m = SmallRLGen9Beta()
m.gin_overrides = None
print(m.get_path_to_checkpoint(m.default_checkpoint))
EOF
```

### 7. Install bot dependencies

All bot dependencies are covered by the Metamon install. See `requirements.txt` for
the full pinned freeze.

## Pokemon Showdown symlink (Metamon submodule)

Metamon's submodule `server/pokemon-showdown` uses an SSH URL and will be empty on clone.
Symlink the already-cloned Showdown repo:

```bash
cd metamon
git submodule deinit -f server/pokemon-showdown
rm -rf .git/modules/server/pokemon-showdown server/pokemon-showdown
ln -s /absolute/path/to/pokemon-showdown server/pokemon-showdown
cd ..
```

## Verify the stack

```bash
source metamon-env/bin/activate
python scripts/verify_phase1.py
```

Expected output: `PHASE 1 OK — MaxDamageBot N/10 wins vs RandomPlayer`.
