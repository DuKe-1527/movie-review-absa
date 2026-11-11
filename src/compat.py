# -*- coding: utf-8 -*-
"""
第三方库兼容垫片

问题：`textrank4zh 0.3` 内部调用 `networkx.from_numpy_matrix`，
      而该接口在 networkx 3.0 中已被移除（本环境 networkx 3.4.2），
      导致 TextRank 关键词/句子抽取静默失败（返回空列表）。

处理：在不修改第三方包的前提下，用行为等价的 `from_numpy_array` 补齐旧接口。
"""
from __future__ import annotations

import numpy as np


def patch_networkx_for_textrank() -> bool:
    """为 textrank4zh 恢复 networkx<3 的 from_numpy_matrix 接口。"""
    import networkx as nx

    if hasattr(nx, "from_numpy_matrix"):
        return False  # 旧版 networkx，无需处理

    def from_numpy_matrix(A, parallel_edges=False, create_using=None):
        """等价于 networkx<3 的 nx.from_numpy_matrix（默认边权重属性为 'weight'）。"""
        return nx.from_numpy_array(np.asarray(A), parallel_edges=parallel_edges,
                                   create_using=create_using, edge_attr="weight")

    nx.from_numpy_matrix = from_numpy_matrix
    return True
