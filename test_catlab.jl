using Catlab
using Catlab.Graphs

println("Catlab.jlのロード完了！")

# 非常にシンプルなグラフ（圏のモックアップ）を作成
g = Graph()
add_vertices!(g, 3)
add_edges!(g, [1, 2], [2, 3])

println("頂点(Object)の数: ", nv(g))
println("辺(Morphism)の数: ", ne(g))
println("応用圏論のシミュレーション環境が正常に機能しています。")