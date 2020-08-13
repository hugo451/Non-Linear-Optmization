import sympy
import numdifftools as nd
import numpy as np


def inaccurate_linear_search(z, n, f, gradf, d):
    """
    z = ponto inicial
    n = multiplicador
    f = função
    gradf = gradienteda função
    d = direção de descida
    t = passo

    """
    t = 0

    while(f(z+t*d) >= f(z) + n*t*np.dot(np.transpose(gradf(z)), d)):
       t = t + 0.001
    return t

def accurate_linear_search(z, f, d):
    """
    z = ponto inicial
    f = função
    d = direção de descida
    t = passo

    """

    t = 0

    while(f(z+t*d) >= f(z)):
        t = t + 0.001
    return t

class NonLinearOptimization:

    def __init__(self, z, f, n = 1, dim = 2):
        """
        z = ponto inicial
        n = multiplicador (busca inexata) (0-1)
        f = função
        d = direção de descida
        dim = dimenção do domínio

        """
        
        self.z = z
        self.f = f
        self.gradf = nd.Gradient(f)
        self.n = n
        self.dim = dim
        self.e = 0


    def basic_algorithm(self):
        
        while np.linalg.norm(self.gradf(self.z)) > self.e + 0.01:
            d = -self.gradf(self.z)
            t = accurate_linear_search(self.z, self.f, d)
            self.z = self.z + t*d
        
        return self.z


    def gradient_descent(self):

        while np.linalg.norm(self.gradf(self.z)) > self.e+ 0.001:
            d = -self.gradf(self.z)
            t = accurate_linear_search(self.z, self.f, d)
            self.z = self.z + t*d
        
        return self.z


    def newtons_metod(self):

        h = nd.Hessian(self.f)

        while np.linalg.norm(self.gradf(self.z)) > self.e + 0.001:
            d = -np.linalg.inv(h(self.z))*self.gradf(self.z)
            t = accurate_linear_search(self.z, self.f, d)
            self.z = self.z + t*d
        
        return self.z
    
    def conjugated_gradient(self):

        d = -self.gradf(self.z)
        i = 0

        while np.linalg.norm(self.gradf(self.z)) > self.e + 0.001:

            t = accurate_linear_search(self.z, self.f, d)

            if i+1 % 2 != 0:
                d = -self.gradf(self.z + t*d) + (self.gradf(self.z + t*d).T*self.gradf(self.z + t*d)/self.gradf(self.z).T*self.gradf(self.z))*d
            else:
                d = - self.gradf(self.z + t*d)
            
            self.z = self.z + t*d

            i = i + 1
        
        return self.z
