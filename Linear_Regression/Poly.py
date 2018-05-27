
"""
Polynomial Class
"""
class Poly():
    def __init__(self,a,k):
        self.a=a
        self.k=k
        self.A =[lambda x,a=a,k=k:[a*n**k for n in x] for a,k in zip(a,k)]
        self.V=None
    def fit(self,x):
        self.V=[fun(x) for fun in self.A]
    def evaluate(self):
        assert self.V != None, "No data to evaluate, please fit\
        data using the poly.fit() method"
        return [sum(i) for i in zip(*self.V)] 
    def poly_features(self):
        assert self.V != None, "No data to generate matrix, please fit\
        data using the poly.fit() method"
        return list(map(list,zip(*self.V)))                       