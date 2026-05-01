#!/usr/bin/env bash
# リポジトリルートで「Python venv + Julia(juliaup)」を一緒に使えるよう PATH を揃える。
# 使い方（bash）:
#   cd /path/to/physics-gnn-surrogate
#   source scripts/setup_env.sh
#
# 初回のみ venv 作成（uv 推奨）:
#   uv venv .venv
#   source .venv/bin/activate
#   uv pip install -r requirements.txt
# そのあと上の source scripts/setup_env.sh でも可

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return
cd "$_REPO_ROOT" || return

# juliaup / julia（venv には含まれない — システム／juliaup 側）
if [[ -d "$HOME/.juliaup/bin" ]]; then
  case ":$PATH:" in
    *:"$HOME/.juliaup/bin":*) ;;
    *) export PATH="$HOME/.juliaup/bin:$PATH" ;;
  esac
fi

# Python venv
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
  echo "[hetero-surrogate] venv: .venv (active)"
else
  echo "[hetero-surrogate] ヒント: uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt" >&2
fi

if command -v julia &>/dev/null; then
  echo "[hetero-surrogate] julia: $(command -v julia) ($(julia --version 2>/dev/null | head -1))"
else
  echo "[hetero-surrogate] 警告: julia が PATH にありません。juliaup 導入と docs/julia_setup.md を参照" >&2
fi

unset _REPO_ROOT
