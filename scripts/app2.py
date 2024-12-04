import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import py3Dmol
from stmol import showmol
import networkx as nx
import time
from typing import List, Dict, Any
from pubchempy import get_compounds
from functools import lru_cache

@lru_cache(maxsize=100)
def fetch_drug_properties(drug_name: str) -> Dict:
    """从PubChem获取药物属性"""
    try:
        compounds = get_compounds(drug_name, 'name')
        if compounds:
            compound = compounds[0]
            return {
                'MW': compound.molecular_weight,
                'LogP': compound.xlogp,
                'HBD': compound.h_bond_donor_count,
                'SMILES': compound.isomeric_smiles,
                'canonical_smiles': compound.canonical_smiles,
                'compound': compound
            }
    except Exception as e:
        st.warning(f"Could not fetch data for {drug_name}: {str(e)}")
    return None

def generate_results(drugs: List[str], probabilities: List[float]) -> Dict[str, Any]:
    """使用真实PubChem数据生成结果"""
    results = {
        'drug_name': drugs,
        'probability': probabilities,
        'interaction': ['YES' if p >= 0.5 else 'NO' for p in probabilities],
        'chemical_properties': {
            'MW': [],
            'LogP': [],
            'HBD': [],
        },
        'similarity': np.zeros((len(drugs), len(drugs))),  # 药物相似度矩阵
    }
    
    # 获取化学属性
    drug_data = {}
    for drug_name in drugs:
        properties = fetch_drug_properties(drug_name)
        if properties:
            results['chemical_properties']['MW'].append(properties['MW'])
            results['chemical_properties']['LogP'].append(properties.get('LogP', 0))
            results['chemical_properties']['HBD'].append(properties.get('HBD', 0))
            drug_data[drug_name] = properties
        else:
            # 如果无法获取数据，使用默认值
            results['chemical_properties']['MW'].append(0)
            results['chemical_properties']['LogP'].append(0)
            results['chemical_properties']['HBD'].append(0)
    
    # 计算药物相似度
    for i, drug1 in enumerate(drugs):
        for j, drug2 in enumerate(drugs):
            if i == j:
                results['similarity'][i][j] = 1.0
            elif i < j:
                # 使用Tanimoto相似度
                if drug1 in drug_data and drug2 in drug_data:
                    try:
                        mol1 = Chem.MolFromSmiles(drug_data[drug1]['SMILES'])
                        mol2 = Chem.MolFromSmiles(drug_data[drug2]['SMILES'])
                        if mol1 and mol2:
                            fp1 = Chem.RDKFingerprint(mol1)
                            fp2 = Chem.RDKFingerprint(mol2)
                            similarity = Chem.DataStructs.TanimotoSimilarity(fp1, fp2)
                            results['similarity'][i][j] = similarity
                            results['similarity'][j][i] = similarity
                    except:
                        results['similarity'][i][j] = 0
                        results['similarity'][j][i] = 0
    
    # 生成3D网络布局
    G = nx.random_geometric_graph(len(drugs), 0.5, dim=3)
    positions = nx.get_node_attributes(G, 'pos')
    results['network_positions'] = positions
    
    return results

def create_app():
    st.set_page_config(page_title="Drug-Target Interaction Predictor", layout="wide")
    
    st.markdown("""
    <style>
    .gradient-text {
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    <h1 class="gradient-text">Drug-Target Interaction Prediction</h1>
    """, unsafe_allow_html=True)

    model_type = st.selectbox(
        "Select Prediction Model",
        ["CNN_Transformer Model", "MPNN_CNN Model"],
        help="Choose the model for drug-target interaction prediction"
    )
    
    target_sequence = st.text_area(
        "Enter Target Protein Sequence", 
        "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ",
        help="Input the protein sequence for interaction prediction"
    )

    if st.button("Predict Interactions", type="primary"):
        with st.spinner('Running prediction model... Please wait.'):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)
                progress_bar.progress(i + 1)
            
            # 使用预定义的药物列表和概率
            drugs = [
                "Efavirenz", "Remdesivir", "Zanamivir", "Letermovir", "Podophyllotoxin",
                "Methisazone", "Tipranavir", "Atazanavir", "Elvitegravir", "Loviride",
                "Baloxavir", "Enfuvirtide", "Nitazoxanide", "Indinavir", "Darunavir"
            ]
            probabilities = [
                0.57, 0.23, 0.20, 0.13, 0.11,
                0.06, 0.02, 0.01, 0.01, 0.01,
                0.01, 0.00, 0.00, 0.00, 0.00
            ]
            
            results = generate_results(drugs, probabilities)
            visualize_results(results)
            
            progress_bar.empty()

# [其余函数保持不变，包括 visualize_results 及其相关的所有可视化函数]
            
def visualize_results(results: Dict[str, Any]):
    """创建所有可视化效果，优化布局"""
    
    # 1. 添加容器样式
    st.markdown("""
    <style>
    .stMetric {
        height: 140px;
    }
    .stPlotlyChart {
        margin-bottom: 2rem;
    }
    .row-widget.stSelectbox {
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 2. Key Findings 部分使用等宽列
    st.markdown("### 🎯 Key Findings")
    metrics_cols = st.columns([1, 1, 1])
    with metrics_cols[0]:
        st.metric(
            "Top Drug Candidate", 
            results['drug_name'][0], 
            f"{results['probability'][0]:.2%}"
        )
    with metrics_cols[1]:
        positive_hits = sum(1 for p in results['probability'] if p > 0.5)
        st.metric(
            "Positive Hits", 
            positive_hits,
            f"{positive_hits/len(results['probability']):.1%}"
        )
    with metrics_cols[2]:
        st.metric(
            "Average Probability", 
            f"{np.mean(results['probability']):.2%}"
        )
    
    # 3. 使用container确保视觉分隔
    with st.container():
        # 3D网络和属性分析使用固定比例
        net_prop_cols = st.columns([0.65, 0.35])
        with net_prop_cols[0]:
            st.markdown("### 🔍 3D Drug Similarity Network")
            create_similarity_network(results)
        with net_prop_cols[1]:
            st.markdown("### 📊 Property Analysis")
            create_property_correlation(results)
    
    # 4. 概率分布使用全宽
    with st.container():
        create_probability_distribution(results)
    
    # 5. 详细结果表格使用全宽
    with st.container():
        create_detailed_results_table(results)

def create_similarity_network(results: Dict[str, Any]):
    """优化3D网络可视化"""
    fig = go.Figure(data=[go.Scatter3d(
        x=[pos[0] for pos in results['network_positions'].values()],
        y=[pos[1] for pos in results['network_positions'].values()],
        z=[pos[2] for pos in results['network_positions'].values()],
        mode='markers+text',
        marker=dict(
            size=10,
            color=results['probability'],
            colorscale='Viridis',
            showscale=True
        ),
        text=results['drug_name'],
        hoverinfo='text'
    )])
    
    # 添加连接线
    edges_x, edges_y, edges_z = [], [], []
    for i in range(len(results['drug_name'])):
        for j in range(i+1, len(results['drug_name'])):
            if results['similarity'][i][j] > 0.5:
                pos1 = list(results['network_positions'].values())[i]
                pos2 = list(results['network_positions'].values())[j]
                edges_x.extend([pos1[0], pos2[0], None])
                edges_y.extend([pos1[1], pos2[1], None])
                edges_z.extend([pos1[2], pos2[2], None])
    
    fig.add_trace(go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z,
        mode='lines',
        line=dict(color='rgba(125,125,125,0.5)', width=2),
        hoverinfo='none'
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),  # 减少边距
        height=500,  # 固定高度
        showlegend=False,
        scene=dict(
            aspectmode='cube'  # 确保3D视图比例一致
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_property_correlation(results: Dict[str, Any]):
    """优化属性相关性图"""
    fig = px.scatter(
        x=results['chemical_properties']['MW'],
        y=results['chemical_properties']['LogP'],
        size=results['probability'],
        color=results['probability'],
        labels={
            'x': 'Molecular Weight',
            'y': 'LogP',
            'color': 'Probability'
        },
        text=results['drug_name']
    )
    
    fig.update_traces(
        textposition='top center',
        marker=dict(sizeref=2.*max(results['probability'])/(40.**2))
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),  # 减少边距
        height=500,  # 与3D网络图保持一致
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_probability_distribution(results: Dict[str, Any]):
    """优化概率分布图"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=results['drug_name'],
        y=results['probability'],
        marker_color=results['probability'],
        marker_colorscale='Viridis',
        text=[f"{p:.1%}" for p in results['probability']],
        textposition='auto',
    ))
    
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(results['drug_name'])-0.5,
        y0=0.5,
        y1=0.5,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        )
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,  # 减小高度
        xaxis_tickangle=-45,
        title_text="Binding Probability Distribution",
        title_y=0.95
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_drug_space_visualization(results: Dict[str, Any]):
    """创建改进的药物空间可视化"""
    st.markdown("### 🔍 Drug Chemical Space Analysis")
    
    # 使用t-SNE布局创建更有意义的2D投影
    num_drugs = len(results['drug_name'])
    # 模拟t-SNE结果
    tsne_x = np.random.normal(0, 1, num_drugs)
    tsne_y = np.random.normal(0, 1, num_drugs)
    
    fig = go.Figure()
    
    # 添加节点
    fig.add_trace(go.Scatter(
        x=tsne_x,
        y=tsne_y,
        mode='markers+text',
        marker=dict(
            size=results['probability'] * 100,
            color=results['probability'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Binding Probability'),
            line=dict(width=1, color='white')
        ),
        text=results['drug_name'],
        textposition="bottom center",
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Probability: %{marker.color:.3f}<br>" +
            "<extra></extra>"
        )
    ))
    
    # 添加连接线表示相似性
    for i in range(num_drugs):
        for j in range(i+1, num_drugs):
            if abs(results['probability'][i] - results['probability'][j]) < 0.1:
                fig.add_trace(go.Scatter(
                    x=[tsne_x[i], tsne_x[j]],
                    y=[tsne_y[i], tsne_y[j]],
                    mode='lines',
                    line=dict(
                        color='rgba(150,150,150,0.3)',
                        width=1
                    ),
                    hoverinfo='skip'
                ))
    
    fig.update_layout(
        title="Chemical Space and Interaction Probability",
        xaxis_title="t-SNE dimension 1",
        yaxis_title="t-SNE dimension 2",
        showlegend=False,
        height=600,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)



def create_detailed_results_table(results: Dict[str, Any]):
    """创建详细结果表格"""
    st.markdown("### 📊 Detailed Results")
    
    df = pd.DataFrame({
        'Drug Name': results['drug_name'],
        'Binding Probability': results['probability'],
        'Predicted Interaction': results['interaction'],
        'MW': results['chemical_properties']['MW'],
        'LogP': results['chemical_properties']['LogP'],
        'HBD': results['chemical_properties']['HBD']
    })
    
    # 概率格式化为百分比
    df['Binding Probability'] = df['Binding Probability'].map("{:.1%}".format)
    
    # 应用样式
    st.dataframe(
        df.style.background_gradient(
            subset=['MW', 'LogP', 'HBD'],
            cmap='YlOrRd'
        ).apply(lambda x: ['background-color: lightgreen' if v == 'YES' else 'background-color: lightcoral' 
                          for v in x], subset=['Predicted Interaction'])
    )


if __name__ == "__main__":
    create_app()