---
title: "サロゲートモデル構築(3)：マルチフィジックスにおける「圏論」の圧倒的優位性の証明"
type: "tech"
topics: ["応用圏論", "Catlab", "HeteroGNN", "マルチフィジックス", "PyTorch Geometric"]
published: false
---

# サロゲートモデル構築(3)：マルチフィジックスにおける「圏論」の圧倒的優位性の証明

## 1. はじめに

現実の機械・構造シミュレーションは、理想バネだけでは足りない。**粘性ダンパー**（速度に比例する抵抗力）、接触、非線形バネなど、**異なる物理法則が辺や要素の種類として混在**する**マルチフィジックス**である。離散モデルでは、それに相当して**複数種類の相互作用**がグラフ上の異なるエッジタイプとして現れる。

本稿では、位置に依存するバネ項と、速度に依存するダンパー項が**別エッジ集合**として存在する系を考え、

- **同質 GNN**：すべての辺を一つの `edge_index` に混ぜ、`GCNConv` で区別なく処理する。
- **圏論的スキーマに対応する Hetero GNN**：`HeteroData` 上で `('node','spring','node')` と `('node','damper','node')` を分離し、`HeteroConv` で**辺タイプごとに独立した畳み込み**を行う。

の学習挙動を比較する。実装の参照先は `demo2_category_multiphysics.py` と出力画像 `hetero_loss_comparison.png` である。

## 2. 検証3：バネとダンパーが混在する系の学習

各ノード $i$ に位置 $u_i$ と速度 $v_i$ を成分とする状態を載せる。バネで結ばれた無向ペア $(i,j)$ には、自然長 0 のフックの法則に従い、ノード $i$ に与える力の寄与を

$$
F^{(\mathrm{spring})}_{i \leftarrow (i,j)} = k\,(u_j - u_i)
$$

のように書く（$j$ 側には反対向きの寄与が入る）。ダンパーで結ばれたペアでは、速度差に比例する項として

$$
F^{(\mathrm{damper})}_{i \leftarrow (i,j)} = c\,(v_j - v_i)
$$

を用いる。全ノードの合力から加速度 $a_i = F_i / m$、オイラー法で次ステップを

$$
u_i(t+\Delta t) = u_i(t) + v_i(t)\,\Delta t,\qquad
v_i(t+\Delta t) = v_i(t) + a_i(t)\,\Delta t
$$

と定め、これを教師 $y$ とする。

```python
# プレースホルダ：物理ターゲット（実装は demo2_category_multiphysics.multiphysics_next_state）
# y = multiphysics_next_state(x, spring_pairs, damper_pairs, k=k, c=c, m=m, dt=dt)
```

## 3. 通常のGNNの限界

同質 GCN は、各辺 $(s \to t)$ について**同じ重み**でメッセージを生成し集約する。バネとダンパーを**同一の `edge_index` に連結**すると、モデルは「どの辺が変位に応じ、どの辺が速度差に応じるか」を**パラメータだけで区別**しなければならない。しかしメッセージの形が辺タイプで本質的に異なるため、単一の同質畳み込みでは**情報が混同**しやすく、損失が**プラトー**（下げ止まり）に達しやすい。

これはハイパーパラメータ調整だけの問題ではなく、**表現クラスが相互作用の型を分けていない**ことに起因する。

## 4. 解決策：応用圏論（Catlab.jl）による厳密なスキーマ設計

応用圏論の言葉でいえば、**オブジェクト**（質点）と、**異なる morphism の族**（バネによる結合、ダンパーによる結合）を**型レベルで別エッジタイプ**として宣言する。Catlab の ACSet（C-Sets）では、このような**スキーマに従うデータ**を一貫して操作でき、Julia 側のグラフ生成・JSON エクスポートと組み合わせやすい。

PyG では、それに対応する表現が **`HeteroData`** である。

```text
(node) --spring--> (node)
(node) --damper--> (node)
```

```python
# プレースホルダ：HeteroData の骨格
# data = HeteroData()
# data["node"].x = x
# data["node", "spring", "node"].edge_index = ei_spring
# data["node", "damper", "node"].edge_index = ei_damper
```

`HeteroConv` で辺タイプごとに `GCNConv` を割り当てれば、**バネ用チャネルとダンパー用チャネルがパラメータ上も分離**し、混同が緩む。

## 5. 検証結果：圏論的Hetero GNNの収束

`demo2_category_multiphysics.py` では、固定プール上の平均 MSE を 200 エポック追跡し、`hetero_loss_comparison.png` に保存している。**シアンの実線**が Hetero（スキーマ分離）側、**赤の点線**が同質 GNN 側である。同一オプティマイザ・学習率のもとで、**Hetero 側の損失がより低い水準へ収束**し、同質側は高いプラトーに留まりやすい傾向が観察される。

> 図：`hetero_loss_comparison.png`（リポジトリ直下。実行して生成する）

再現手順のプレースホルダ：

```bash
# .venv/bin/python demo2_category_multiphysics.py
# 出力: hetero_loss_comparison.png
```

## 6. おわりに

真に現場で使える **Physics-Informed AI** には、単に損失項に物理を足すだけでなく、**相互作用の種類・対称性・結合トポロジをデータ構造として固定する層**が必要である。応用圏論によるスキーマ設計は、その**型安全な骨格**を与え、PyG の **Hetero** 機構は実装側の対応物となる。

技術顧問、PoC 設計、Julia–Python ブリッジを含むアーキテクチャ構築については、**YOUTRUST** 等のプロフィール経由で相談を受け付けている。本連載で示した検証を踏まえ、具体的なシミュレータ連携やサロゲート要件に応じた設計を一緒に詰めていきたい。

---

- 第1回：`docs/01_simple_spring_and_bridge.md`（固定 $N$ の罠と JSON ブリッジ）  
- 第2回：`docs/02_scale_generalization.md`（$N$ 変動と MLP の破綻）  
- 第3回：本稿（マルチフィジックスと Hetero）
