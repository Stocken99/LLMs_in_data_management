import torch
torch.set_printoptions(precision=2,sci_mode=False)

#embeddings = torch.tensor(
#    [[1.2, 3.2, .12],
#     [-2.1, 2.3, 1]])

embeddings = torch.tensor(
    [[0.0,0.0,1.0],   # Boy
     [0.0,1.0,0.0],   # Bites
     [1.0,0.0,0.0]])  # Dog

K_W = torch.tensor(
    [[0.2, 1.3, .2],
    [.4, .5, -.4],
    [-.2, 1, -.33]])

Q_W = torch.tensor(
    [[0.23, 1.13, .27],
    [.47, .54, -.46],
    [-.22, 1, -.36]])

V_W = torch.tensor(
    [[1.0, 0, 0],
     [0, 1.0, 0],
     [0, 0, 1.0]])

O_W = torch.tensor(
    [[.1, 0, 0],
     [0, .1, 0],
     [0, 0, .1]])

K = embeddings.mm(K_W)
Q = embeddings.mm(Q_W)
V = embeddings.mm(V_W)

M = Q.mm(K.t())
S = M.softmax(dim=1)

A = S.mm(V)

Delta = A.mm(O_W)

New = embeddings + Delta

normalized_vectors = New.norm(dim=1, keepdim=True)
New_normalized = New / normalized_vectors

print("Weights:")
print(f"Q_W:\n{Q_W}")
print(f"K_W:\n{K_W}")
print(f"V_W:\n{V_W}")
print(f"O_W:\n{O_W}")

print("Computation:")
print(f"Input Embeddings:\n{embeddings}")
print(f"Q:\n{Q}")
print(f"K:\n{K}")
print(f"V:\n{V}")
print(f"M:\n{M}")
print(f"S:\n{S}")
print(f"A:\n{A}")
print(f"Delta:\n{New}")
print(f"Output Embeddings:\n{New_normalized}")






