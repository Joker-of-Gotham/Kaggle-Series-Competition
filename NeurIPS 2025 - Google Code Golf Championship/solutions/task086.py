import collections

def p(g):
    h, w = len(g), len(g[0])
    o = [[0] * w for _ in range(h)]
    visited = [[False] * w for _ in range(h)]

    for r_start in range(h):
        for c_start in range(w):
            if g[r_start][c_start] != 0 and not visited[r_start][c_start]:
                # 1. 使用BFS寻找一个完整的对象及其属性
                q = collections.deque([(r_start, c_start)])
                visited[r_start][c_start] = True
                component_cells = []
                colors = collections.defaultdict(int)
                min_r, max_r = r_start, r_start
                min_c, max_c = c_start, c_start

                while q:
                    r, c = q.popleft()
                    component_cells.append((r, c))
                    min_r, max_r = min(min_r, r), max(max_r, r)
                    min_c, max_c = min(min_c, c), max(max_c, c)
                    colors[g[r][c]] += 1
                    
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                
                # 2. 分析对象结构
                obj_h = max_r - min_r + 1
                obj_w = max_c - min_c + 1
                
                if len(colors) < 2: continue

                c_outer = max(colors, key=colors.get)
                c_inner = min(colors, key=colors.get)

                inner_min_r, inner_max_r = h, -1
                inner_min_c, inner_max_c = w, -1
                for r_cell, c_cell in component_cells:
                    if g[r_cell][c_cell] == c_inner:
                        inner_min_r = min(inner_min_r, r_cell)
                        inner_max_r = max(inner_max_r, r_cell)
                        inner_min_c = min(inner_min_c, c_cell)
                        inner_max_c = max(inner_max_c, c_cell)
                
                h_in = inner_max_r - inner_min_r + 1
                w_in = inner_max_c - inner_min_c + 1

                # 3. 创建颜色交换后的中心部分
                s_orig = [[g[min_r + i][min_c + j] for j in range(obj_w)] for i in range(obj_h)]
                s_swap = [[0] * obj_w for _ in range(obj_h)]
                for i in range(obj_h):
                    for j in range(obj_w):
                        if s_orig[i][j] == c_outer: s_swap[i][j] = c_inner
                        elif s_orig[i][j] == c_inner: s_swap[i][j] = c_outer

                # 4. 将转换后的图形绘制到输出网格
                paste_r_start = min_r - h_in
                paste_c_start = min_c - w_in
                h_new = obj_h + 2 * h_in
                w_new = obj_w + 2 * w_in

                for i in range(h_new):
                    for j in range(w_new):
                        out_r, out_c = paste_r_start + i, paste_c_start + j
                        if 0 <= out_r < h and 0 <= out_c < w:
                            is_center = (h_in <= i < h_in + obj_h) and (w_in <= j < w_in + obj_w)
                            is_in_plus = (h_in <= i < h_in + obj_h) or (w_in <= j < w_in + obj_w)

                            if is_in_plus:
                                if is_center:
                                    color = s_swap[i - h_in][j - w_in]
                                    if color != 0: o[out_r][out_c] = color
                                else:
                                    o[out_r][out_c] = c_outer
    return o