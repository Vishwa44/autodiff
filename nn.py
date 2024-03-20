import numpy as np


class Tensor:
  def __init__(self, data, _children=(), _op='', label = "", requires_grad = False):
    self.data = data # numpy ndarray
    self.children = [] # [tensor]

    self.grad = np.zeros_like(data) # init grad to zeros
    self._backward = lambda: None # pointer to the backward function
    self._prev = set(_children) # set of children tensors
    self._op = _op # operation on tensor
    self.label = label # tensor label
    self.requires_grad = requires_grad # to differentiate or not


  def shape(self):
    return self.data.shape

  def __repr__(self):
    return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

  def data(self):
    ''' Returns the data stored in the tensor as a Numpy Array. '''
    return self.data

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return Add(self, other)

  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return Mul(self, other)

  def __rmul__(self, other):
    return self * other

  def __neg__(self):
    return Neg(self)

  def __sub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return self + (-other)

  def __rsub__(self, other):
    return self - other

  def __matmul__(self, other):
    return Matmul(self, other)

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    return Pow(self, other)

  def __truediv__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return Div(self, other)

  def exp(self):
    return Exp(self)

  def backward(self): # differetiate
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = np.ones_like(self.data)
    print(topo)
    for node in reversed(topo):
      node._backward()


def Add(a, b):
    requires_grad = a.requires_grad or b.requires_grad

    # Get new Tensor's data:
    data = a.data + b.data

    # Create new Tensor:
    z = Tensor(data, (a, b), '+', requires_grad=requires_grad)

    def _backward():
      if a.requires_grad:
        a.grad += z.grad
      if b.requires_grad:
        b.grad += z.grad
    z._backward = _backward

    return z

def Mul(a, b):
    requires_grad = a.requires_grad or b.requires_grad

    # Get new Tensor's data:
    data = a.data * b.data

    # Create new Tensor:
    z = Tensor(data, (a, b), '*', requires_grad=requires_grad)

    def _backward():
      if a.requires_grad:
        a.grad += b.data * z.grad
      if b.requires_grad:
        b.grad += a.grad * z.grad
    z._backward = _backward

    return z

def Neg(a):
    # Get new Tensor's data:
    data = a.data * -1

    # Create new Tensor:
    z = Tensor(data, (a,), 'neg', requires_grad=a.requires_grad)

    def _backward():
      if a.requires_grad:
        a.grad += -1*z.grad
    z._backward = _backward

    return z

def Matmul(a, b):
    requires_grad = a.requires_grad or b.requires_grad
    # Get new Tensor's data:
    data = a.data @ b.data

    # Create new Tensor:
    z = Tensor(data, (a, b), '@', requires_grad=requires_grad)

    def _backward():
      if a.requires_grad:
        a.grad += z.grad @ b.data.T
      if b.requires_grad:
        b.grad += a.data.T @ z.grad
    z._backward = _backward

    return z

def Pow(a, b):
    # Get new Tensor's data:

    data = np.power(a.data, b)
    # Create new Tensor:
    z = Tensor(data, (a, ), 'pow', requires_grad = a.requires_grad)

    def _backward():
      if a.requires_grad:
        a.grad += b * (np.power(a.data, (b - 1))) * z.grad
    z._backward = _backward

    return z

def Div(a, b):
    requires_grad = a.requires_grad or b.requires_grad

    # Get new Tensor's data:
    data = a.data / b.data

    # Create new Tensor:
    z = Tensor(data, (a, b), '*', requires_grad=requires_grad)
    def _backward():
      if a.requires_grad:
        a.grad += z.grad * (1/b.data)
      if b.requires_grad:
        b.grad += - z.grad * a.data/(b.data ** 2)
    z._backward = _backward
    return z

def Exp(a):
    data = np.exp(a.data)
    z = Tensor(data, (a, ), 'exp', requires_grad=a.requires_grad)
    def _backward():
        if a.requires_grad:
          a.grad += (z.data) * z.grad
    z._backward = _backward
    return z