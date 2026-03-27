"""
MatrixVis - LaTeX导出模块
生成专业的LaTeX报告和PDF
"""

import numpy as np
from typing import Dict, List, Optional
import json
from datetime import datetime

def generate_latex_report(result: Dict) -> str:
    """
    生成LaTeX格式的计算报告

    Args:
        result: 计算结果字典

    Returns:
        LaTeX源代码字符串
    """
    matrix = result.get('matrix', np.array([]))
    calc_type = result.get('type', '未知运算')
    timestamp = result.get('timestamp', datetime.now())

    latex_parts = []

    # 文档开头
    latex_parts.append(r"""\documentclass[12pt,a4paper]{article}""")
    latex_parts.append(r"""\usepackage[UTF8]{ctex}""")
    latex_parts.append(r"""\usepackage{amsmath,amssymb,amsthm}""")
    latex_parts.append(r"""\usepackage{geometry}""")
    latex_parts.append(r"""\usepackage{booktabs}""")
    latex_parts.append(r"""\usepackage{xcolor}""")
    latex_parts.append(r"""\usepackage{listings}""")
    latex_parts.append(r"""\usepackage{hyperref}""")
    latex_parts.append(r"""\geometry{margin=2.5cm}""")
    latex_parts.append(r"""\title{\textbf{MatrixVis 计算报告}}""")
    latex_parts.append(r"""\author{线性代数AI可视化求解系统}""")
    latex_parts.append(r"""\date{""" + timestamp.strftime('%Y年%m月%d日 %H:%M:%S') + r"""}""")
    latex_parts.append(r"""\begin{document}""")
    latex_parts.append(r"""\maketitle""")
    latex_parts.append(r"""\section{输入矩阵}""")

    # 添加矩阵
    latex_parts.append(matrix_to_latex(matrix, "A"))

    # 矩阵属性
    rank_val = np.linalg.matrix_rank(matrix)
    latex_parts.append(f"""
    \\subsection{{矩阵属性}}
    \\begin{{itemize}}
        \\item 维度: ${matrix.shape[0]} \\times {matrix.shape[1]}$
        \\item 秩: {rank_val}
    """)

    if matrix.shape[0] == matrix.shape[1]:
        try:
            cond = np.linalg.cond(matrix)
            latex_parts.append(f"    \\item 条件数: $\\kappa(A) = {cond:.4e}$\n")
        except:
            pass

    latex_parts.append(r"""\end{itemize}

\section{计算结果}
""")

    # 行列式
    if 'determinant' in result:
        det = result['determinant']
        latex_parts.append(r"""
\subsection{行列式}
""")
        if 'value' in det:
            latex_parts.append(f"$$\\det(A) = {det['value']:.6f}$$\n\n")

        if 'steps' in det and det['steps']:
            latex_parts.append(r"""\subsubsection{LU分解步骤}
""")
            for i, step in enumerate(det['steps'][:5]):  # 最多显示5步
                desc = step.get('description', f'步骤 {i+1}')
                latex_parts.append(f"\\textbf{{步骤 {i+1}}}: {desc}\n\n")

                if 'matrix' in step:
                    latex_parts.append(matrix_to_latex(step['matrix'], f"A_{{{i+1}}}"))

    # 逆矩阵
    if 'inverse' in result and result['inverse']:
        inv = result['inverse']
        latex_parts.append(r"""
\subsection{逆矩阵}
""")
        if 'matrix' in inv:
            latex_parts.append(matrix_to_latex(inv['matrix'], "A^{-1}"))

            # 验证
            latex_parts.append(r"""
\subsubsection{验证}
$AA^{-1} = I$:
""")
            verification = matrix @ inv['matrix']
            latex_parts.append(matrix_to_latex(verification, "AA^{-1}"))

    # 特征值
    if 'eigenvalues' in result and result['eigenvalues']:
        eigen = result['eigenvalues']
        latex_parts.append(r"""
\subsection{特征值与特征向量}
""")
        if 'values' in eigen:
            values = eigen['values']
            latex_parts.append(r"""\begin{align*}
""")
            for i, val in enumerate(values):
                latex_parts.append(f"\\lambda_{{{i+1}}} &= {val:.6f} \\\\")
            latex_parts.append(r"""\end{align*}
""")

        if 'vectors' in eigen:
            latex_parts.append(r"""
\subsubsection{特征向量}
""")
            vectors = eigen['vectors']
            for i in range(min(3, len(vectors))):
                latex_parts.append(f"\\textbf{{对应}} $\\lambda_{{{i+1}}}$:\n")
                latex_parts.append(vector_to_latex(vectors[:, i], f"v_{{{i+1}}}"))

    # 线性方程组
    if 'solution' in result:
        sol = result['solution']
        latex_parts.append(r"""
\subsection{线性方程组求解}
""")
        if 'x' in sol and sol['x'] is not None:
            latex_parts.append(vector_to_latex(sol['x'], "x"))

        if 'type' in sol:
            latex_parts.append(f"\\textbf{{解的情况}}: {sol['type']}\n\n")

    # 结尾
    latex_parts.append(r"""\section{总结}

本报告由 \textbf{MatrixVis} 系统自动生成。

\vspace{1cm}
\begin{center}
\textcolor{gray}{\small 海南师范大学参赛团队 | 2025中国大学生计算机设计大赛}
\end{center}

\end{document}
""")

    return "".join(latex_parts)

def matrix_to_latex(matrix: np.ndarray, name: str = "A") -> str:
    """
    将矩阵转换为LaTeX格式

    Args:
        matrix: 输入矩阵
        name: 矩阵名称

    Returns:
        LaTeX字符串
    """
    if matrix is None or len(matrix) == 0:
        return ""

    rows, cols = matrix.shape if len(matrix.shape) == 2 else (len(matrix), 1)

    latex = f"$${name} = \\begin{{bmatrix}}\n"

    for i in range(rows):
        row_str = ""
        for j in range(cols if len(matrix.shape) == 2 else 1):
            if len(matrix.shape) == 2:
                val = matrix[i, j]
            else:
                val = matrix[i]

            # 格式化数值
            if abs(val) < 1e-10:
                row_str += "0"
            elif abs(val - int(val)) < 1e-10:
                row_str += f"{int(val)}"
            else:
                row_str += f"{val:.4f}"

            if j < (cols if len(matrix.shape) == 2 else 1) - 1:
                row_str += " & "

        latex += row_str
        if i < rows - 1:
            latex += " \\\\"

    latex += "\n\\end{bmatrix}$$\n\n"

    return latex

def vector_to_latex(vector: np.ndarray, name: str = "v") -> str:
    """
    将向量转换为LaTeX格式

    Args:
        vector: 输入向量
        name: 向量名称

    Returns:
        LaTeX字符串
    """
    return matrix_to_latex(vector.reshape(-1, 1), name)

def generate_pdf_report(result: Dict) -> bytes:
    """
    生成PDF报告（需要LaTeX环境）

    Args:
        result: 计算结果字典

    Returns:
        PDF文件字节
    """
    # 注意：这需要系统安装LaTeX环境
    # 实际使用时可以使用pdflatex或xelatex编译

    latex_content = generate_latex_report(result)

    # 这里返回LaTeX源码，实际PDF生成需要外部工具
    return latex_content.encode('utf-8')

def generate_json_export(result: Dict) -> str:
    """
    生成JSON格式的导出数据

    Args:
        result: 计算结果字典

    Returns:
        JSON字符串
    """
    # 转换numpy数组为列表
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_result = convert_to_serializable(result)

    return json.dumps(serializable_result, indent=2, ensure_ascii=False)

def generate_markdown_report(result: Dict) -> str:
    """
    生成Markdown格式的报告

    Args:
        result: 计算结果字典

    Returns:
        Markdown字符串
    """
    matrix = result.get('matrix', np.array([]))
    calc_type = result.get('type', '未知运算')
    timestamp = result.get('timestamp', datetime.now())

    md = f"""# MatrixVis 计算报告

**生成时间**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

---

## 输入矩阵

```
{np.array2string(matrix, precision=4, suppress_small=True)}
```

**维度**: {matrix.shape[0]}×{matrix.shape[1]}

**秩**: {np.linalg.matrix_rank(matrix)}

---

## 计算结果

"""

    # 行列式
    if 'determinant' in result:
        det = result['determinant']
        md += f"""### 行列式

**结果**: det(A) = {det.get('value', 'N/A'):.6f}

"""

    # 逆矩阵
    if 'inverse' in result and result['inverse']:
        inv = result['inverse']
        md += f"""### 逆矩阵

```
{np.array2string(inv.get('matrix', []), precision=4, suppress_small=True)}
```

"""

    # 特征值
    if 'eigenvalues' in result and result['eigenvalues']:
        eigen = result['eigenvalues']
        md += f"""### 特征值

"""
        if 'values' in eigen:
            for i, val in enumerate(eigen['values']):
                md += f"- λ_{i+1} = {val:.6f}\n"
        md += "\n"

    md += """---

*本报告由 MatrixVis 系统自动生成*

海南师范大学参赛团队 | 2025中国大学生计算机设计大赛
"""

    return md

def generate_csv_export(matrix: np.ndarray) -> str:
    """
    生成CSV格式的矩阵数据

    Args:
        matrix: 输入矩阵

    Returns:
        CSV字符串
    """
    if matrix is None or len(matrix) == 0:
        return ""

    lines = []
    for row in matrix:
        line = ",".join([str(val) for val in row])
        lines.append(line)

    return "\n".join(lines)
