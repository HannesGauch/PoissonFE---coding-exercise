# programme to solve Poisson equation using FE

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# input parameters
l_x = 1 # domain size
l_y = 1
nel_x = 30 # number of elements in x
nel_y = 30 # number of elements in y



# compute mesh size
nel = nel_x*nel_y
dx = l_x/nel_x
dy = l_y/nel_y

nnodes = (nel_x+1)*(nel_y+1)


#create nodes
node_coords = np.zeros((nnodes,2))
free_node = np.ones((nnodes,),dtype=int)
n_bnd_nodes = 0

for i in range(nel_x+1):
    for j in range(nel_y+1):
        node_coords[i*(nel_y+1)+j] = [i*dx,j*dy]
        #check if node is boundary node
        if i==0 or i==nel_x or j==0 or j==nel_y:
            free_node[i*(nel_y+1)+j] = 0
            n_bnd_nodes += 1
        
#create elements
el_nodes = np.zeros((nel,4),dtype=int)

for i in range(nel_x):
    for j in range(nel_y):
        #element numberung starting with smallest x,y value, counterclockwise
        el_nodes[i*nel_y+j] = [i*(nel_y+1)+j, (i+1)*(nel_y+1)+j, (i+1)*(nel_y+1)+j+1, i*(nel_y+1)+j+1]


#define shape function and gradient of shape functions of 4-node quadrilateral elements

def N_el(xi,eta):
    N_mat = np.zeros((1,4))
    N_mat[0,0] = 1/4*(1-xi)*(1-eta)
    N_mat[0,1] = 1/4*(1+xi)*(1-eta)
    N_mat[0,2] = 1/4*(1+xi)*(1+eta)
    N_mat[0,3] = 1/4*(1-xi)*(1+eta)
    return N_mat
    
def B_el(xi,eta):
    B_mat = np.zeros((2,4))
    B_mat[0,0] = dy*(eta-1)
    B_mat[0,1] = dy*(1-eta)
    B_mat[0,2] = dy*(1+eta)
    B_mat[0,3] = dy*(-1-eta)
    B_mat[1,0] = dx*(xi-1)
    B_mat[1,1] = dx*(-1-xi)
    B_mat[1,2] = dx*(1+xi)
    B_mat[1,3] = dx*(1-xi)
    return B_mat/(2*dx*dy)
    
# compute element stiffness matrix using Gauss Quadrature
ngp = 2
gauss_l = [+1/np.sqrt(3),-1/np.sqrt(3)]
gauss_w = [1.0,1.0]
det_Je = dx*dy/4 # determinant of element Jacobian

K_el = np.zeros((4,4))
for i in range(ngp):
    for j in range(ngp):
        K_el += gauss_w[i]*gauss_w[j]*det_Je*np.matmul(np.transpose(B_el(gauss_l[i],gauss_l[j])),B_el(gauss_l[i],gauss_l[j]))
    
# assemble global stiffness matrix using direct assembly
K_glob = np.zeros((nnodes,nnodes))   

for i in range(nel):
    for j in range(4):
        for k in range(4):
            node1 = el_nodes[i,j]
            node2 = el_nodes[i,k]
            K_glob[node1,node2] += K_el[j,k]
            
# compute global source term matrix
f_glob = np.zeros((nnodes,))

def f_func(x,y):
    return 4*(-y**2+y)*np.sin(np.pi*x)
    #return 1.0
    
for i in range(nel):
    # compute element source term matrix
    xe = node_coords[el_nodes[i],0]
    ye = node_coords[el_nodes[i],1]
    f_el = np.zeros((4,1))
    for j in range(ngp):
        for k in range(ngp):
            f_el += gauss_w[j]*gauss_w[k]*det_Je*np.transpose(N_el(gauss_l[j],gauss_l[k]))*f_func(np.matmul(N_el(gauss_l[j],gauss_l[k]),xe),np.matmul(N_el(gauss_l[j],gauss_l[k]),ye))
    # assemble global source term matrix
    for l in range(4):
        node = el_nodes[i,l]
        f_glob[node] += f_el[l]


    
# partition the system in boundary nodes and free nodes
K_red = np.zeros((nnodes-n_bnd_nodes,nnodes-n_bnd_nodes))
f_red = np.zeros((nnodes-n_bnd_nodes,))

# look for free nodes and copy matrix entries to reduced matrices
idxi = 0
for i in range(nnodes):
    if free_node[i]:
        f_red[idxi] = f_glob[i]
        idxj = 0
        for j in range(nnodes):
            if free_node[j]:        
                K_red[idxi,idxj] = K_glob[i,j]
                idxj += 1
        idxi +=1
        
# solve the partitioned system

u_red = np.linalg.solve(K_red,f_red)

# reassemble system

u_glob = np.zeros((nnodes,))
u_glob[free_node==1] = u_red


#plot results

fig = plt.figure()
ax = fig.gca(projection='3d')

surf=ax.plot_trisurf(node_coords[:,0], node_coords[:,1], u_glob, linewidth=0.2, antialiased=True,cmap=plt.cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()