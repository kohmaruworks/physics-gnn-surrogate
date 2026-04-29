#!/usr/bin/env bash
# act / basic と同様: リポルートで「Python venv + Julia(juliaup)」を一緒に使えるよう PATH を揃える。
# 使い方（bash）:
#   cd /path/to/physics-gnn-surrogate-phase2
#   source scripts/phase2_env.sh
#
# 初回のみ venv 作成（act / basic と同様に uv 推奨）:
#   uv venv .venv
#   source .venv/bin/activate
#   uv pip install -r requirements.txt
# そのあと上の source scripts/phase2_env.sh でも可

_P2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return
cd "$_P2_ROOT" || return

# juliaup / julia（venv には含まれない — システム／juliaup 側）
if [[ -d "$HOME/.juliaup/bin" ]]; then
  case ":$PATH:" in
    *:"$HOME/.juliaup/bin":*) ;;
    *) export PATH="$HOME/.juliaup/bin:$PATH" ;;
  esac
fi

# Python venv（act / basic の .venv と同じ考え方）
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
  echo "[phase2] venv: .venv (active)"
else
  echo "[phase2] ヒント: uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt" >&2
fi

if command -v julia &>/dev/null; then
  echo "[phase2] julia: $(command -v julia) ($(julia --version 2>/dev/null | head -1))"
else
  echo "[phase2] 警告: julia が PATH にありません。juliaup 導入と docs/julia/julia_setup.md を参照" >&2
fi

unset _P2_ROOT
