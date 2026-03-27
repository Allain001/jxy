"""
MatrixVis - 可视化模块
基于Plotly的矩阵运算可视化
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import pandas as pd

def plot_matrix_heatmap(states: List[np.ndarray], title: str = "矩阵变化过程") -> go.Figure:
    """
    绘制矩阵变化热力图
    
    Args:
        states: 矩阵状态列表
        title: 图表标题
        
    Returns:
        Plotly图形
    """
    if not states:
        return go.Figure()
    
    # 创建子图
    n_states = min(len(states), 6)  # 最多显示6个状态
    cols = min(3, n_states)
    rows = (n_states + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"步骤 {i}" for i in range(n_states)],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    # 计算统一的颜色范围
    all_values = np.concatenate([s.flatten() for s in states[:n_states]])
    zmin, zmax = np.min(all_values), np.max(all_values)
    
    for i, state in enumerate(states[:n_states]):
        row = i // cols + 1
        col = i % cols + 1
        
        heatmap = go.Heatmap(
            z=state,
            colorscale='RdBu',
            zmid=0,
            zmin=zmin,
            zmax=zmax,
            showscale=(i == 0),
            text=np.round(state, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        )
        
        fig.add_trace(heatmap, row=row, col=col)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=300 * rows,
        showlegend=False
    )
    
    return fig

def plot_lu_animation(L: np.ndarray, U: np.ndarray, steps: List[Dict]) -> go.Figure:
    """
    绘制LU分解动画帧
    
    Args:
        L: 下三角矩阵
        U: 上三角矩阵
        steps: 分解步骤
        
    Returns:
        Plotly图形
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['L (下三角)', 'U (上三角)', 'A = LU'],
        horizontal_spacing=0.1
    )
    
    # L矩阵
    fig.add_trace(
        go.Heatmap(
            z=L,
            colorscale='Blues',
            showscale=False,
            text=np.round(L, 2),
            texttemplate='%{text}',
            name='L'
        ),
        row=1, col=1
    )
    
    # U矩阵
    fig.add_trace(
        go.Heatmap(
            z=U,
            colorscale='Reds',
            showscale=False,
            text=np.round(U, 2),
            texttemplate='%{text}',
            name='U'
        ),
        row=1, col=2
    )
    
    # 验证 A = LU
    A_reconstructed = L @ U
    fig.add_trace(
        go.Heatmap(
            z=A_reconstructed,
            colorscale='Viridis',
            showscale=False,
            text=np.round(A_reconstructed, 2),
            texttemplate='%{text}',
            name='A = LU'
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        title=dict(text='LU分解结果', font=dict(size=16)),
        height=400
    )
    
    return fig

def plot_gauss_jordan_animation(states: List[np.ndarray]) -> go.Figure:
    """
    绘制高斯-约当消元动画
    
    Args:
        states: 增广矩阵状态列表
        
    Returns:
        Plotly图形
    """
    if not states:
        return go.Figure()
    
    # 创建动画帧
    frames = []
    for i, state in enumerate(states):
        n = state.shape[1] // 2
        
        frame = go.Frame(
            data=[
                go.Heatmap(
                    z=state[:, :n],
                    colorscale='Blues',
                    showscale=False,
                    name='A',
                    x=[f'a{i+1}' for i in range(n)],
                    y=[f'row{i+1}' for i in range(n)]
                ),
                go.Heatmap(
                    z=state[:, n:],
                    colorscale='Reds',
                    showscale=False,
                    name='I',
                    x=[f'e{i+1}' for i in range(n)],
                    y=[f'row{i+1}' for i in range(n)]
                )
            ],
            name=f'frame{i}'
        )
        frames.append(frame)
    
    # 初始状态
    n = states[0].shape[1] // 2
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=states[0][:, :n],
                colorscale='Blues',
                showscale=False,
                name='A'
            ),
            go.Heatmap(
                z=states[0][:, n:],
                colorscale='Reds',
                showscale=False,
                name='I'
            )
        ],
        frames=frames
    )
    
    # 添加动画控制
    fig.update_layout(
        title=dict(text='高斯-约当消元过程', font=dict(size=16)),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶️ 播放',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
                },
                {
                    'label': '⏸️ 暂停',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
                }
            ]
        }],
        height=400
    )
    
    return fig

def plot_eigenvalue_geometry(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> go.Figure:
    """
    绘制特征值几何解释（2D/3D）
    
    Args:
        eigenvalues: 特征值数组
        eigenvectors: 特征向量矩阵
        
    Returns:
        Plotly图形
    """
    n = len(eigenvalues)
    
    if n == 2:
        # 2D可视化
        fig = go.Figure()
        
        # 绘制单位圆
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=x_circle, y=y_circle,
            mode='lines',
            name='单位圆',
            line=dict(color='gray', dash='dash')
        ))
        
        # 绘制特征向量
        colors = ['red', 'blue']
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            # 归一化特征向量
            vec = vec / np.linalg.norm(vec)
            
            # 原始向量
            fig.add_trace(go.Scatter(
                x=[0, vec[0]],
                y=[0, vec[1]],
                mode='lines+markers',
                name=f'v{i+1} (λ={val:.2f})',
                line=dict(color=colors[i], width=3),
                marker=dict(size=8)
            ))
            
            # 变换后的向量
            fig.add_trace(go.Scatter(
                x=[0, val*vec[0]],
                y=[0, val*vec[1]],
                mode='lines+markers',
                name=f'Av{i+1} = {val:.2f}v{i+1}',
                line=dict(color=colors[i], width=2, dash='dot'),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=dict(text='特征值的几何意义（2D）', font=dict(size=16)),
            xaxis=dict(title='x', range=[-3, 3], zeroline=True),
            yaxis=dict(title='y', range=[-3, 3], zeroline=True, scaleanchor='x'),
            height=500,
            showlegend=True
        )
        
    else:
        # 3D可视化（简化版）
        fig = go.Figure()
        
        # 绘制椭球
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        
        # 使用特征值缩放
        if len(eigenvalues) >= 3:
            a, b, c = np.abs(eigenvalues[:3])
        else:
            a = b = c = 1
        
        x = a * np.outer(np.cos(u), np.sin(v))
        y = b * np.outer(np.sin(u), np.sin(v))
        z = c * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale='Viridis',
            opacity=0.7,
            name='变换后的单位球'
        ))
        
        # 添加坐标轴
        for i, (val, color) in enumerate(zip(eigenvalues[:3], ['red', 'green', 'blue'])):
            vec = np.zeros(3)
            vec[i] = val
            fig.add_trace(go.Scatter3d(
                x=[0, vec[0]],
                y=[0, vec[1]],
                z=[0, vec[2]],
                mode='lines+markers',
                name=f'λ{i+1} = {val:.2f}',
                line=dict(color=color, width=4),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=dict(text='特征值的几何意义（3D）', font=dict(size=16)),
            scene=dict(
                xaxis=dict(range=[-3, 3]),
                yaxis=dict(range=[-3, 3]),
                zaxis=dict(range=[-3, 3]),
                aspectmode='cube'
            ),
            height=600
        )
    
    return fig

def plot_convergence_curve(errors: List[float], title: str = "收敛曲线") -> go.Figure:
    """
    绘制迭代算法的收敛曲线
    
    Args:
        errors: 误差列表
        title: 图表标题
        
    Returns:
        Plotly图形
    """
    iterations = list(range(1, len(errors) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=errors,
        mode='lines+markers',
        name='误差',
        line=dict(color='#2196F3', width=2),
        marker=dict(size=6)
    ))
    
    # 添加收敛阈值线
    fig.add_hline(
        y=1e-10,
        line_dash="dash",
        line_color="red",
        annotation_text="收敛阈值 (1e-10)"
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title='迭代次数', type='log'),
        yaxis=dict(title='误差 (对数刻度)', type='log'),
        height=400
    )
    
    return fig

def plot_matrix_comparison(matrices: Dict[str, np.ndarray], title: str = "矩阵对比") -> go.Figure:
    """
    对比多个矩阵
    
    Args:
        matrices: 矩阵字典 {名称: 矩阵}
        title: 图表标题
        
    Returns:
        Plotly图形
    """
    n = len(matrices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=list(matrices.keys()),
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    # 统一颜色范围
    all_values = np.concatenate([m.flatten() for m in matrices.values()])
    zmin, zmax = np.min(all_values), np.max(all_values)
    
    for i, (name, matrix) in enumerate(matrices.items()):
        row = i // cols + 1
        col = i % cols + 1
        
        heatmap = go.Heatmap(
            z=matrix,
            colorscale='RdBu',
            zmid=0,
            zmin=zmin,
            zmax=zmax,
            showscale=(i == 0),
            text=np.round(matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        )
        
        fig.add_trace(heatmap, row=row, col=col)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=300 * rows
    )
    
    return fig

def plot_svd_visualization(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> go.Figure:
    """
    绘制SVD分解可视化
    
    Args:
        U: 左奇异向量矩阵
        S: 奇异值对角矩阵
        Vt: 右奇异向量转置
        
    Returns:
        Plotly图形
    """
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=['U', 'Σ', 'Vᵀ', 'A = UΣVᵀ'],
        horizontal_spacing=0.05
    )
    
    # U
    fig.add_trace(
        go.Heatmap(z=U, colorscale='Blues', showscale=False, name='U'),
        row=1, col=1
    )
    
    # Σ
    fig.add_trace(
        go.Heatmap(z=S, colorscale='Reds', showscale=False, name='Σ'),
        row=1, col=2
    )
    
    # Vt
    fig.add_trace(
        go.Heatmap(z=Vt, colorscale='Greens', showscale=False, name='Vᵀ'),
        row=1, col=3
    )
    
    # 重构
    A_reconstructed = U @ S @ Vt
    fig.add_trace(
        go.Heatmap(z=A_reconstructed, colorscale='Viridis', showscale=False, name='A'),
        row=1, col=4
    )
    
    fig.update_layout(
        title=dict(text='奇异值分解 (SVD)', font=dict(size=16)),
        height=400
    )
    
    return fig

def create_animated_matrix_evolution(states: List[np.ndarray], interval: int = 500) -> go.Figure:
    """
    创建矩阵演化的动画
    
    Args:
        states: 矩阵状态列表
        interval: 帧间隔（毫秒）
        
    Returns:
        Plotly动画图形
    """
    if not states:
        return go.Figure()
    
    # 创建帧
    frames = []
    for i, state in enumerate(states):
        frame = go.Frame(
            data=[go.Heatmap(
                z=state,
                colorscale='RdBu',
                zmid=0,
                text=np.round(state, 2),
                texttemplate='%{text}'
            )],
            name=f'frame{i}'
        )
        frames.append(frame)
    
    # 初始状态
    fig = go.Figure(
        data=[go.Heatmap(
            z=states[0],
            colorscale='RdBu',
            zmid=0,
            text=np.round(states[0], 2),
            texttemplate='%{text}'
        )],
        frames=frames
    )
    
    # 添加滑块
    sliders = [{
        'steps': [
            {
                'args': [[f'frame{k}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                'label': f'{k}',
                'method': 'animate'
            }
            for k in range(len(states))
        ],
        'transition': {'duration': 0},
        'x': 0.1,
        'y': 0,
        'currentvalue': {'visible': True, 'prefix': '步骤: '}
    }]
    
    fig.update_layout(
        title=dict(text='矩阵演化过程', font=dict(size=16)),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶️ 播放',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': interval, 'redraw': True}, 'fromcurrent': True}]
                },
                {
                    'label': '⏸️ 暂停',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
                }
            ]
        }],
        sliders=sliders,
        height=500
    )
    
    return fig
