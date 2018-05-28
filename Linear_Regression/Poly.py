
"""
Polynomial Class
"""
class Poly():
    def __init__(self,a,k):
        self.a=a
        self.k=k
        self.A =lambda x,a=a,k=k:[[a*n**k for a,k in zip(a,k)] for n in x] 
        self.V=None
    def fit(self,x):
        self.V=self.A(x)
    def evaluate(self):
        assert self.V != None, "No data to evaluate, please fit\
        data using the poly.fit() method"
        return [sum(i) for i in self.V] 
    def poly_features(self):
        assert self.V != None, "No data to generate matrix, please fit\
        data using the poly.fit() method"
        return self.V                       