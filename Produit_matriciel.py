# Fonction faisant le produit matriciel de deux matrices

def produit_matriciel(A,B):
    """Fonction faisant le produit matriciel de deux matrices"""
    n = len(A)
    p = len(B[0])
    C = [[0 for j in range(p)] for i in range(n)]
    for i in range(n): # Pour chaque ligne de A
        for j in range(p): # Pour chaque colonne de B
            for k in range(len(B)): # Pour chaque colonne de A
                C[i][j] += A[i][k]*B[k][j] #
    return C

# Test
A = [[1,2,3],[4,5,6]]
B = [[1,2],[3,4],[5,6]]
print(produit_matriciel(A,B))
print(produit_matriciel(B,A))

A=[[1],[2],[3]]
B=[[1,2,3]]
print(produit_matriciel(A,B))
print(produit_matriciel(B,A))
