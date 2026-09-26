#!/usr/bin/env bash
#SBATCH -J clipC_square
#SBATCH -p msismall
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24gb
#SBATCH -t 03:00:00
#SBATCH --tmp 20gb
#SBATCH -o /users/1/munda057/SignWay/render_square_%j.log
set -euo pipefail
cd ~/SignWay
source /scratch.global/munda057/venvs/jmem/bin/activate     # venv only; no conda on top
which python
bash render_square.sh