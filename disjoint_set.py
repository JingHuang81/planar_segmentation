class DisjointSet():
    def __init__(self, size):
        self.ids = range(size)
    
    def array(self):
        return [self.root(i) for i in self.ids]

    def root(self, i):
        while i != self.ids[i]:
            # Path compression.
            self.ids[i] = self.ids[self.ids[i]]
            i = self.ids[i]
        return i

    def find(self, p, q):
        return self.root(p) == self.root(q)

    def unite(self, p, q):
        self.ids[self.root(q)] = self.root(p)
