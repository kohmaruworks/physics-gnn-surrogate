# ホストに Julia を入れずに、このリポの Pkg 環境を使う例
#   docker build -t hetero-surrogate-julia .
#   docker run --rm -v "$PWD":/app -w /app hetero-surrogate-julia

FROM julia:1.10-bookworm

WORKDIR /app
COPY . .

RUN julia --project=src/julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

CMD ["julia", "--project=src/julia", "src/julia/03_simulation.jl"]
