#!/usr/bin/env python
import os
import scanpy as sc
import pandas as pd
import numpy as np
import harmonypy as hm
import anndata
from sklearn.metrics import silhouette_score
import clustree
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

sc.settings.verbosity = 3
sc.logging.print_header()

#图片保存设置
sc.settings.figdir = 'figures'
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc.settings.autoshow = False

# 读取所有 GSM 样本
base_dir = "samples"
adata_list = []
sample_names = []  # 新增：记录样本名

for sample_dir in sorted(os.listdir(base_dir)):
    if sample_dir.startswith("GSM"):
        sample_path = os.path.join(base_dir, sample_dir)
        print(f"读取: {sample_path}")
        
        # 读取每个样本
        adata = sc.read_10x_mtx(
            sample_path,
            var_names='gene_symbols',
            make_unique=True,
            cache=True  # 添加 cache 加速
        )
        
        # 添加样本信息
        adata.obs['sample'] = sample_dir
        adata_list.append(adata)
        sample_names.append(sample_dir)  # 新增：记录样本名

# 合并所有样本（keys 需要和 adata_list 长度一致）
if len(adata_list) > 1:
    adata = sc.concat(adata_list, join='outer', keys=sample_names)  # 修改这里
    print(f"合并后: {adata}")
else:
    adata = adata_list[0] if adata_list else None

if adata is None:
    print("没有读取到任何数据！")
    exit(1)

print(adata)

#-----------------质控------------------
# 计算线粒体基因比例（人类基因通常以 MT- 开头）
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, inplace=True)

# 绘制小提琴图（QC 前）
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True, save='_qc_before.png')
print(f"过滤前: {adata.n_obs} 个细胞, {adata.n_vars} 个基因")

sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='pct_counts_mt', save='_qc_genes_vs_umi_color.png')

# 过滤细胞
sc.pp.filter_cells(adata, min_genes=200)      # 每个细胞至少 200 个基因
sc.pp.filter_genes(adata, min_cells=3)        # 每个基因至少在 3 个细胞中表达

# 过滤线粒体基因比例过高（>20%）
adata = adata[adata.obs.pct_counts_mt < 20, :]

# 查看 total_counts 的分布
print(adata.obs['total_counts'].describe())

#基于分位数（更灵活）该样本双细胞十分明显
upper_limit = adata.obs['total_counts'].quantile(0.95)
print(f"过滤阈值: {upper_limit}")
adata = adata[adata.obs.total_counts < upper_limit, :]

print(adata)

#绘制小提琴图（QC 后）
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True, save='_qc_after.png')

# 相关关系散点图
sc.pl.scatter(adata, 'total_counts', 'n_genes_by_counts', color='pct_counts_mt', save='_qc_scatter.png')

#--------------标准化--------------------
# 标准化
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

# 显示表达量最高的20个基因
sc.pl.highest_expr_genes(adata, n_top=20, show=True,save='_top20_genes.png')  # 修正：添加 show=True
adata.raw = adata

#根据输出自行判断使用哪种
# 标记高变基因（准备降维）
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='sample')
print(f"高变基因数: {sum(adata.var.highly_variable)}")
sc.pl.highly_variable_genes(adata, save='_hvg_n_top_2000.png')

# 计算
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# 绘制特异性基因散点图
print(f"min_mean 方法高变基因数: {sum(adata.var.highly_variable)}")
sc.pl.highly_variable_genes(adata, save='_hvg_min_mean.png')

#----------------------获取特异基因------------------------------
# 获取只有特异性基因的数据集
adata = adata[:, adata.var.highly_variable]
# 回归每个细胞的总计数和表达的线粒体基因的百分比的影响。
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
# 将每个基因缩放到单位方差。阈值超过标准偏差 10。
#sc.pp.scale(adata, max_value=10)

# 绘制 PCA 图
sc.tl.pca(adata, svd_solver='arpack')
print("X_pca shape:", adata.obsm['X_pca'].shape)
sc.pl.pca(adata, color='CST3', save='_pca_CST3.png')
sc.pl.pca(adata, color='sample', save='_pca_sample.png')

# 绘制方差解释率
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, save='_pca_variance_ratio.png')
cumsum = np.cumsum(adata.uns['pca']['variance_ratio'])
n_pcs = np.argmax(cumsum >= 0.9) + 1
print(f"达到 90% 累积方差解释率需要 {n_pcs} 个主成分")

#------------------------消除批次效应----------------------------------
# 运行 Harmony
ho = hm.run_harmony(
    adata.obsm['X_pca'],
    adata.obs,
    vars_use=['sample'],
    max_iter_harmony=20,
    random_state=0
)

# 转置后保存得加 .T
adata.obsm['X_pca_harmony'] = ho.Z_corr
#print(f"X_pca_harmony shape: {adata.obsm['X_pca_harmony'].shape}")
#维度不对
#sc.external.pp.harmony_integrate(adata, key='sample', basis='X_pca')
print("Harmony 校正完成！")

# 使用 embedding 函数绘制 Harmony 校正后的 PCA
sc.pl.embedding(adata, basis='X_pca_harmony', color='CST3', save='_pca_after_harmony_CST3.png')
sc.pl.embedding(adata, basis='X_pca_harmony', color='sample', save='_pca_after_harmony_sample.png')

# 绘制方差解释率
sc.pl.pca_variance_ratio(adata, n_pcs=40, log=True, save='_pca_after_harmony_variance_ratio.png')
cumsum = np.cumsum(adata.uns['pca']['variance_ratio'])
n_pcs = np.argmax(cumsum >= 0.9) + 1
print(f"达到 90% 累积方差解释率需要 {n_pcs} 个主成分")

# 将每个基因缩放到单位方差。阈值超过标准偏差 10。
sc.pp.scale(adata, max_value=10)
n_pcs = 30  # Harmony 后通常直接指定 30-50 个 PCs
print(f"批次校正完成，后续分析将使用 {n_pcs} 个主成分")

#--------------------聚类-----------------------------------------------------
# 使用 Harmony 校正后的主成分构建邻居图
sc.pp.neighbors(adata, n_neighbors=40, use_rep='X_pca_harmony', n_pcs=n_pcs)
sc.tl.umap(adata)
sc.pl.umap(adata, color='sample', save='_umap_after_harmony_sample.png')
sc.pl.umap(adata, color='CST3', save='_umap_after_harmony_CST3.png')
sc.tl.tsne(adata, use_rep='X_pca_harmony', n_pcs=n_pcs )
sc.pl.tsne(adata, color='sample',save='_tsne_sample.png')
sc.pl.tsne(adata, color='CST3',save='_tsne_CST3.png')

#leiden图聚类
# 计算
sc.tl.leiden(adata,random_state=42)
# 绘制
sc.pl.umap(adata, color=['leiden'],save='_leiden.png')


#检索标记基因
#methon(t-test\wilcoxon\logreg)
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test',use_raw=True)
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False,save='_rank.png')

# 提取排名靠前的基因
result = adata.uns["rank_genes_groups"]
groups = result["names"].dtype.names
marker_genes_df = pd.DataFrame({
group + "_" + key[:1]: result[key][group]
for group in groups for key in ["names", "pvals"]
})
print(marker_genes_df.head())

anndata.settings.allow_write_nullable_strings = True
adata.write('adata_processed.h5ad')
print("数据已保存为 adata_processed.h5ad")

# ==================== 1. 读取数据 ====================
print("读取数据...")
adata = sc.read_h5ad("adata_processed.h5ad")
print(f"数据形状: {adata.shape}")
print(f"细胞数: {adata.n_obs}, 基因数: {adata.n_vars}")

# ==================== 2. 检查Harmony结果 ====================
print("\n检查Harmony结果...")

# 检查是否存在Harmony结果
if 'X_pca_harmony' not in adata.obsm:
    print("⚠️ 未找到Harmony结果，请先运行Harmony整合")
    print("📌 可用的降维结果:")
    for key in adata.obsm.keys():
        print(f"  - {key}")
    
    # 如果存在X_pca，使用PCA替代
    if 'X_pca' in adata.obsm:
        print("\n使用PCA结果替代Harmony...")
        adata.obsm['X_pca_harmony'] = adata.obsm['X_pca'].copy()
    else:
        print("❌ 未找到合适的降维结果，请先运行PCA")
        exit()
else:
    print("✅ Harmony结果已存在")

# ==================== 3. 构建邻居图 ====================
print("\n构建邻居图...")
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=15, use_rep='X_pca_harmony', metric='euclidean')
print("✅ 邻居图构建完成")

# ==================== 4. 多分辨率聚类 ====================
resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
print("\n执行多分辨率聚类...")

cluster_counts = {}
for res in resolutions:
    cluster_key = f'leiden_res_{res}'
    sc.tl.leiden(adata, resolution=res, key_added=cluster_key,random_state=42)
    n_clusters = adata.obs[cluster_key].nunique()
    cluster_counts[res] = n_clusters
    print(f"  分辨率 {res}: {n_clusters} 个聚类")

# ==================== 5. 准备clustree数据 ====================
print("\n准备clustree数据...")
clustering_data = pd.DataFrame()

# 对分辨率进行排序
sorted_resolutions = sorted(resolutions)

# 创建连续的整数列名（clustree要求）
for i, res in enumerate(sorted_resolutions, 1):
    col_name = f'leiden_res_{i}'
    # 获取数据并转换为整数
    data = adata.obs[f'leiden_res_{res}'].values
    
    # 如果是分类类型，转换为整数
    if pd.api.types.is_categorical_dtype(data):
        data = data.astype(str).astype(int)
    else:
        data = pd.Series(data).astype(int)
    
    clustering_data[col_name] = data

# 创建分辨率映射
resolution_map = {f'leiden_res_{i}': res for i, res in enumerate(sorted_resolutions, 1)}
print("\n分辨率映射:")
for col, res in resolution_map.items():
    print(f"  {col} -> {res}")


# ==================== 6. 生成clustree图 ====================
print("\n生成clustree图...")

import os
os.makedirs("figures", exist_ok=True)

try:
    import clustree
    import matplotlib.pyplot as plt
    
    # 确保数据是整数类型
    clustering_data_int = clustering_data.astype(int)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    
    # 使用 images 参数（你的版本需要）
    result = clustree.clustree(
        data=clustering_data_int,
        prefix='leiden_res_',
        images=fig,  # ← 关键：传入 fig 对象
        node_color='prefix',
        edge_color='samples'
    )
    
    plt.tight_layout()
    plt.savefig('figures/clustree.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Clustree图已保存为 figures/clustree.png")
    
except Exception as e:
    print(f"⚠️ Clustree库绘制失败: {e}")
    print("📌 使用手动可视化方案...")
    
    # 这里插入下面的手动可视化代码
    
    # ==================== 7. 手动可视化方案 ====================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 子图1：聚类数量柱状图
    n_clusters = [clustering_data[f'leiden_res_{i}'].nunique() for i in range(1, len(sorted_resolutions) + 1)]
    
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(sorted_resolutions)), n_clusters,
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_resolutions))))
    ax1.set_xticks(range(len(sorted_resolutions)))
    ax1.set_xticklabels([f'{res}' for res in sorted_resolutions], rotation=45)
    ax1.set_xlabel('Resolution', fontsize=12)
    ax1.set_ylabel('Number of Clusters', fontsize=12)
    ax1.set_title('Number of Clusters at Different Resolutions', fontsize=14)
    
    for bar, n in zip(bars, n_clusters):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{n}', ha='center', va='bottom', fontsize=10)
    
    # 子图2：分辨率趋势
    ax2 = axes[0, 1]
    ax2.plot(sorted_resolutions, n_clusters, 'o-', linewidth=2, markersize=10, color='darkblue')
    ax2.set_xlabel('Resolution', fontsize=12)
    ax2.set_ylabel('Number of Clusters', fontsize=12)
    ax2.set_title('Resolution vs Number of Clusters', fontsize=14)
    ax2.grid(True, alpha=0.3)
    for res, n in zip(sorted_resolutions, n_clusters):
        ax2.annotate(f'n={n}', (res, n), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    # 子图3：ARI相似性热图
    ax3 = axes[1, 0]
    n_res = len(sorted_resolutions)
    similarity_matrix = np.zeros((n_res, n_res))
    
    for i in range(n_res):
        for j in range(n_res):
            col1 = f'leiden_res_{i+1}'
            col2 = f'leiden_res_{j+1}'
            ari = adjusted_rand_score(clustering_data[col1], clustering_data[col2])
            similarity_matrix[i, j] = ari
    
    im = ax3.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax3.set_xticks(range(n_res))
    ax3.set_yticks(range(n_res))
    ax3.set_xticklabels([f'{res}' for res in sorted_resolutions])
    ax3.set_yticklabels([f'{res}' for res in sorted_resolutions])
    ax3.set_xlabel('Resolution', fontsize=12)
    ax3.set_ylabel('Resolution', fontsize=12)
    ax3.set_title('Clustering Similarity (ARI)', fontsize=14)
    plt.colorbar(im, ax=ax3)
    
    # 子图4：聚类大小分布
    ax4 = axes[1, 1]
    cluster_sizes_data = []
    labels = []
    for i in range(1, len(sorted_resolutions) + 1):
        col = f'leiden_res_{i}'
        sizes = clustering_data[col].value_counts().values
        cluster_sizes_data.append(sizes)
        labels.append(f'{sorted_resolutions[i-1]}')
    
    bp = ax4.boxplot(cluster_sizes_data, labels=labels, patch_artist=True)
    ax4.set_xlabel('Resolution', fontsize=12)
    ax4.set_ylabel('Cluster Size', fontsize=12)
    ax4.set_title('Cluster Size Distribution', fontsize=14)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('figures/clustree.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 聚类分析图已保存为 figures/clustree.png")

# ==================== 8. 保存结果 ====================
print("\n保存结果...")

# 保存聚类标签
for res in sorted_resolutions:
    cluster_key = f'leiden_res_{res}'
    # 添加原始分辨率标签
    adata.obs[f'cluster_res_{res}'] = adata.obs[cluster_key]

# 保存为h5ad文件
adata.write("adata_processed_with_clusters.h5ad")
print("✅ 结果已保存到 adata_processed_with_clusters.h5ad")

# ==================== 9. 显示统计信息 ====================
print("\n📊 聚类统计摘要:")
print("=" * 60)

# 创建统计表
stats_df = pd.DataFrame({
    'Resolution': sorted_resolutions,
    'Number_of_Clusters': n_clusters
})

print("\n分辨率与聚类数量:")
print(stats_df.to_string(index=False))

# 详细统计
print("\n详细聚类统计:")
for i, res in enumerate(sorted_resolutions, 1):
    col = f'leiden_res_{i}'
    n = clustering_data[col].nunique()
    sizes = clustering_data[col].value_counts()
    print(f"\n分辨率 {res}:")
    print(f"  聚类数量: {n}")
    print(f"  聚类大小范围: {sizes.min()} - {sizes.max()}")
    print(f"  平均聚类大小: {sizes.mean():.1f}")
    print(f"  中位聚类大小: {sizes.median():.1f}")

#leiden图聚类
# 计算
best_res = 0.8
sc.tl.leiden(adata,resolution=best_res, key_added='leiden', random_state=42)
# 绘制
sc.pl.umap(adata, color=['leiden'],save='_leiden_after_culster.png')
print("best_res have ending")

#检索标记基因
#methon(t-test\wilcoxon\logreg)
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test', use_raw=True)
sc.pl.rank_genes_groups(adata, n_genes=30, sharey=False,save='_rank_best_res.png')

anndata.settings.allow_write_nullable_strings = True
adata.write('adata_processed.h5ad')
print("✅ 数据已保存为 adata_processed.h5ad")
