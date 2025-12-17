def p(g):
    """
    找到“灰色笼子”，将其中的背景色填充为黄色。

    最终解题思路:
    该算法通过一个两阶段过程来解决问题：
    1.  拓扑分析：首先，通过广度优先搜索(BFS)从图像边界出发，精确地识别出
        所有与外界连通的背景区域，定义为“外部空间”。
    2.  几何验证：然后，找到所有不与“外部空间”连通的、被完全隔离的背景区域。
        对这些独立的“内部区域”进行几何形状验证，只有当一个区域是完美的矩形时，
        才将其填充。

    这个方法能够正确处理所有情况，包括由边界和灰色线条共同围成的笼子，
    同时能准确区分独立的“笼子”和起连接作用的“连接架”。
    """
    rows = len(g)
    if not rows:
        return []
    cols = len(g[0])
    if not cols:
        return [[] for _ in range(rows)]

    output_grid = [row[:] for row in g]

    # 步骤 1: 识别所有与边界连通的“外部空间”的黑色像素
    is_outside = [[False for _ in range(cols)] for _ in range(rows)]
    queue = []

    # 将所有位于边界上的黑色像素作为BFS的起始点
    for r in range(rows):
        if g[r][0] == 0 and not is_outside[r][0]:
            is_outside[r][0] = True
            queue.append((r, 0))
        if cols > 1 and g[r][cols - 1] == 0 and not is_outside[r][cols - 1]:
            is_outside[r][cols - 1] = True
            queue.append((r, cols - 1))

    for c in range(1, cols - 1):
        if g[0][c] == 0 and not is_outside[0][c]:
            is_outside[0][c] = True
            queue.append((0, c))
        if rows > 1 and g[rows - 1][c] == 0 and not is_outside[rows - 1][c]:
            is_outside[rows - 1][c] = True
            queue.append((rows - 1, c))

    # 使用BFS扩展，标记所有外部空间的像素
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and g[nr][nc] == 0 and not is_outside[nr][nc]:
                is_outside[nr][nc] = True
                queue.append((nr, nc))

    # 步骤 2 & 3: 寻找所有内部区域，并验证其为矩形
    visited_internal = [[False for _ in range(cols)] for _ in range(rows)]
    for r_start in range(rows):
        for c_start in range(cols):
            # 寻找一个未被访问过的、不属于外部空间的黑色像素
            if g[r_start][c_start] == 0 and not is_outside[r_start][c_start] and not visited_internal[r_start][c_start]:
                
                # 发现一个新的内部区域，用BFS获取其所有像素
                internal_component = []
                internal_q = [(r_start, c_start)]
                visited_internal[r_start][c_start] = True
                
                min_r, max_r = r_start, r_start
                min_c, max_c = c_start, c_start
                
                head_internal = 0
                while head_internal < len(internal_q):
                    r_int, c_int = internal_q[head_internal]
                    head_internal += 1
                    
                    internal_component.append((r_int, c_int))
                    min_r, max_r = min(min_r, r_int), max(max_r, r_int)
                    min_c, max_c = min(min_c, c_int), max(max_c, c_int)
                    
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r_int + dr, c_int + dc
                        if 0 <= nr < rows and 0 <= nc < cols and \
                           g[nr][nc] == 0 and not is_outside[nr][nc] and \
                           not visited_internal[nr][nc]:
                            visited_internal[nr][nc] = True
                            internal_q.append((nr, nc))
                
                # 步骤 3: 验证该区域是否为完美的矩形
                expected_area = (max_r - min_r + 1) * (max_c - min_c + 1)
                if len(internal_component) == expected_area:
                    # 步骤 4: 如果是，则在输出网格上填充
                    for r_fill, c_fill in internal_component:
                        output_grid[r_fill][c_fill] = 4
                        
    return output_grid