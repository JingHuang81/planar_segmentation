class Dart():
    def __init__(self, h, t, r, l, w):
        self.head = h
        self.tail = t
        self.right = r
        self.left = l
        self.weight = w
    def __str__(self):
        return str((self.head, self.tail, self.right, self.left))


class GridGraph():
    def __init__(self, h, w, weights):
        self.h = h
        self.w = w
        self.weights = weights
        self.primal = {i : {} for i in range(h * w)}
        self.dual = {i : {} for i in range((h - 1) * (w - 1) + 1)}
        self._init_grid()
    
    def _init_grid(self):
        f_outer = (self.h - 1) * (self.w - 1)
        for v in range(self.h * self.w):
            i, j = v // self.w, v % self.w
            f = (i - 1) * (self.w - 1) + j
            f1 = f if v >= self.w and j < self.w - 1 else f_outer
            f2 = f - 1 if v >= self.w and j > 0 else f_outer
            f3 = (f + self.w - 2 if v < self.w * (self.h -1) and j > 0
                                 else f_outer)
            f4 = (f + self.w - 1 if (v < self.w * (self.h - 1) and
                                     j < self.w - 1) else f_outer)
            assert(max(f1, f2, f3, f4) <= f_outer)
            darts_cclkw = [
                (v, v + 1, f1, f4, self.weights[v][v+1]) \
                        if j < self.w - 1 else None,
                (v, v - self.w, f2, f1, self.weights[v][v-self.w]) \
                        if i > 0 else None,
                (v, v - 1, f3, f2, self.weights[v][v-1]) if j > 0 else None,
                (v, v + self.w, f4, f3, self.weights[v][v+self.w]) \
                        if i < self.h - 1 else None]
            darts_cclkw = [d for d in darts_cclkw if d is not None]
            self.primal[v] = {(d[1], d[2], d[3]) : Dart(*d)
                              for d in darts_cclkw}
            for d in self.primal[v].values():
                self.dual[d.right][(d.left, d.tail, d.head)] = Dart(
                        d.right, d.left, d.tail, d.head, d.weight)

    def __str__(self):
        graph_str = 'primal:\n'
        graph_str += '\n'.join([str(v) + ' : ' + ','.join(map(str, sorted(self.primal[v].values()))) for v in self.primal])
        graph_str += '\ndual:\n'
        graph_str += '\n'.join([str(f) + ' : ' + ','.join(map(str, sorted(self.dual[f].values()))) for f in self.dual])
        return graph_str

    def vertices(self):
        return self.primal.keys()

    def edges(self):
        return set([e for e in v.keys() for v in self.primal.values()])

    def contract_edge(self, h, t, r=None, l=None):
        # if r is None, there should not be any double edge in the primal graph.
        if r is None or l is None:
            k_contract = [k for k in self.primal[h] if k[0] == t]
            #assert(len(k_contract) < 2)
            if len(k_contract) == 0:
                return
            r = k_contract[0][1]
            l = k_contract[0][2]
        # Faces which are the endpoints of dual edge.
        f_left = self.primal[h][(t, r, l)].left
        f_right = r
        # Primal-edge contraction and removal of its dual.
        # For all dual vertices f, degree(f) > 2, this is the only update.
        self._contract_and_remove(h, t, r, l, self.primal)
        # Dual-edge contraction and removal of its primal.
        # For a face f, degree(f) <= 2, this cleans up bigon and monogon.
        for f in set([f_left, f_right]):
            if len(self.dual[f]) <= 2 and len(self.dual[f]) > 0:
                # Pick an edge at dual vertex f
                dual_d = self.dual[f].values()[0]
                self._contract_and_remove(
                    dual_d.tail, dual_d.head, dual_d.left, dual_d.right,
                    self.dual)

    def _dual(self, graph):
        return self.dual if graph == self.primal else self.primal

    def _contract_and_remove(self, h, t, r, l, graph):
        #if len(graph) <= 6:
        #    print h, t, r, l
        # Contract edge in G
        d = graph[h].pop((t, r, l))
        d_rev = graph[t].pop((h, d.left, d.right))
        # Remove dual-edge in G*
        dual_d = self._dual(graph)[d.right].pop((d.left, t, h))
        dual_d_rev = self._dual(graph)[d.left].pop((r, h, t))
        for d_del in graph[t].values():
            # Redirect the head/tail of edge from t to h in G.
            graph[d_del.tail][(h, d_del.left, d_del.right)] = \
                    graph[d_del.tail][(t, d_del.left, d_del.right)]
            graph[d_del.tail][(h, d_del.left, d_del.right)].tail = h
            graph[d_del.tail].pop((t, d_del.left, d_del.right))
            graph[h][(d_del.tail, d_del.right, d_del.left)] = d_del
            graph[h][(d_del.tail, d_del.right, d_del.left)].head = h
            # Redirect the right/left of dual-edge from t to h in G*.
            f_left, f_right = d_del.left, d_del.right
            # The right side of dual edge is t not h, since haven't updated yet
            dual_d_del = self._dual(graph)[f_left][(f_right, t, d_del.tail)]
            self._dual(graph)[f_left][(f_right, h, d_del.tail)] = dual_d_del
            self._dual(graph)[f_left][(f_right, h, d_del.tail)].right = h
            self._dual(graph)[f_left].pop((f_right, t, d_del.tail))
            dual_d_del_rev = self._dual(graph)[f_right][(f_left, d_del.tail, t)]
            self._dual(graph)[f_right][(f_left, d_del.tail, h)] = dual_d_del_rev
            self._dual(graph)[f_right][(f_left, d_del.tail, h)].left = h
            self._dual(graph)[f_right].pop((f_left, d_del.tail, t))
        # Remove vertex t.
        if len(graph) > 1:
            graph.pop(t)
        #if len(graph) <= 6:
        #    print self
         

if __name__ == '__main__':
    g = GridGraph(3, 3)
    # H, T, R
    g.contract_edge(3, 4, 0)
    test1 = """
primal:
0 : (0, 1, 4, 0),(0, 3, 0, 4)
1 : (1, 0, 0, 4),(1, 2, 4, 1),(1, 3, 1, 0)
2 : (2, 1, 1, 4),(2, 5, 4, 1)
3 : (3, 0, 4, 0),(3, 5, 1, 3),(3, 6, 2, 4)
5 : (5, 2, 1, 4),(5, 3, 1, 3),(5, 8, 4, 3)
6 : (6, 3, 2, 4),(6, 7, 2, 4)
7 : (7, 3, 2, 3),(7, 6, 4, 2),(7, 8, 3, 4)
8 : (8, 5, 3, 4),(8, 7, 3, 4)
dual:
0 : (0, 1, 1, 3),(0, 4, 0, 1),(0, 4, 3, 0)
1 : (1, 0, 3, 1),(1, 3, 5, 3),(1, 4, 1, 2),(1, 4, 2, 5)
2 : (2, 3, 3, 7),(2, 4, 6, 3),(2, 4, 7, 6)
3 : (3, 1, 3, 5),(3, 2, 7, 4),(3, 4, 5, 8),(3, 4, 8, 7)
4 : (4, 0, 0, 3),(4, 0, 1, 0),(4, 1, 2, 1),(4, 1, 5, 5),(4, 2, 3, 6),(4, 2, 6, 7),(4, 3, 7, 8),(4, 3, 8, 5)
    """
    assert(str(g == test1))

    # Test2
    g.contract_edge(3, 0, 4)
    print g
