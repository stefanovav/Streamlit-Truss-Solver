import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Truss Solver")
st.title("Truss Solver Input Panel")

# =========================================================
# MATERIAL
# =========================================================

st.subheader("Material Properties")
E = st.number_input("Young's Modulus E (N/m²)", value=200e9, format="%.2e")
A = st.number_input("Cross-sectional Area A (m²)", value=0.005, format="%.6f")
xFac = st.number_input("Scale factor for plotted displacements", value=500)

# =========================================================
# NODES
# =========================================================

st.subheader("Nodal Coordinates (x, y)")

default_nodes = np.array([
    [0,6],
    [4,6],
    [8,6],
    [12,6],
    [16,6],
    [12,2],
    [8,0],
    [4,2]
])

nodes_text = st.text_area(
    "Enter nodal coordinates (x, y):",
    value="\n".join([f"{r[0]}, {r[1]}" for r in default_nodes]),
    height=200
)

nodes = np.array([
    [float(x) for x in line.split(",")]
    for line in nodes_text.splitlines()
    if line.strip()
])

numberOfNodes = len(nodes)
nDoF = numberOfNodes * 2

# =========================================================
# MEMBERS
# =========================================================

st.subheader("Members [node_i, node_j]")

default_members = np.array([
    [1,2],[2,3],[3,4],[4,5],
    [5,6],[6,7],[7,8],
    [1,8],[2,8],[3,7],[4,6],[3,8],[3,6]
])

members_text = st.text_area(
    "Enter member connections:",
    value="\n".join([f"{r[0]}, {r[1]}" for r in default_members]),
    height=250
)

members = np.array([
    [int(x) for x in line.split(",")]
    for line in members_text.splitlines()
    if line.strip()
])

# =========================================================
# SUPPORTS
# =========================================================

st.subheader("Supports")

restrained_input = st.text_input(
    "Restrained DOF (start at 1, comma-separated)",
    value="1, 2, 10"
)

restrainedDoF = [int(x.strip()) for x in restrained_input.split(",") if x.strip().isdigit()]
restrainedIndex = [x-1 for x in restrainedDoF if x-1 < nDoF]

# =========================================================
# LOADS
# =========================================================

st.subheader("Loads")

dof_input = st.text_input(
    "DOF to apply forces (start at 0)",
    value="3, 5, 7"
)

force_dofs = [int(x.strip()) for x in dof_input.split(",") if x.strip().isdigit()]

forceVector = np.zeros((nDoF,1))

for dof in force_dofs:
    if dof < nDoF:
        force = st.number_input(f"Force at DOF {dof} (N)", value=0.0)
        forceVector[dof] = force

# =========================================================
# VISUALIZATION
# =========================================================

st.subheader("Structure")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect("equal")

for m in members:
    ni, nj = m
    xi, yi = nodes[ni-1]
    xj, yj = nodes[nj-1]
    ax.plot([xi,xj],[yi,yj],'b')

for i, node in enumerate(nodes):
    ax.plot(node[0],node[1],'bo')
    ax.text(node[0],node[1],str(i+1),color='red')

ax.grid()
st.pyplot(fig)

# =========================================================
# FEM SOLVER
# =========================================================

def memberOrientation(memberNo):
    ni, nj = members[memberNo-1]
    xi, yi = nodes[ni-1]
    xj, yj = nodes[nj-1]
    dx = xj - xi
    dy = yj - yi
    L = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    return theta, L

orientations = []
lengths = []

for i in range(len(members)):
    angle, length = memberOrientation(i+1)
    orientations.append(angle)
    lengths.append(length)

Kp = np.zeros((nDoF,nDoF))

for i, m in enumerate(members):
    ni, nj = m
    theta = orientations[i]
    L = lengths[i]
    c = np.cos(theta)
    s = np.sin(theta)

    k = (E*A/L)*np.array([
        [ c*c, c*s,-c*c,-c*s],
        [ c*s, s*s,-c*s,-s*s],
        [-c*c,-c*s, c*c, c*s],
        [-c*s,-s*s, c*s, s*s]
    ])

    dof = [
        2*(ni-1),2*(ni-1)+1,
        2*(nj-1),2*(nj-1)+1
    ]

    for r in range(4):
        for c2 in range(4):
            Kp[dof[r],dof[c2]] += k[r,c2]

# Reduce system
Ks = np.delete(Kp,restrainedIndex,axis=0)
Ks = np.delete(Ks,restrainedIndex,axis=1)

forceRed = np.delete(forceVector,restrainedIndex,axis=0)

# Solve
Ured = np.linalg.solve(Ks,forceRed)

UG = np.zeros((nDoF,1))
j=0
for i in range(nDoF):
    if i not in restrainedIndex:
        UG[i] = Ured[j]
        j+=1

FG = Kp @ UG

# =========================================================
# RESULTS
# =========================================================

st.subheader("Reactions")
for idx in restrainedIndex:
    st.write(f"Reaction at DOF {idx+1}: {round(FG[idx].item()/1000,2)} kN")

st.subheader("Nodal Displacements")
for i in range(numberOfNodes):
    ux = UG[2*i][0]
    uy = UG[2*i+1][0]
    st.write(f"Node {i+1}: Ux = {ux:.6e} m , Uy = {uy:.6e} m")

st.subheader("Member Forces")

for i, m in enumerate(members):
    ni, nj = m
    theta = orientations[i]
    L = lengths[i]
    c = np.cos(theta)
    s = np.sin(theta)

    dof = [2*(ni-1),2*(ni-1)+1,2*(nj-1),2*(nj-1)+1]
    disp = UG[dof]

    T = np.array([[c,s,0,0],[0,0,c,s]])
    local = T @ disp
    F = (E*A/L)*(local[1]-local[0])
    st.write(f"Member {i+1} ({ni}-{nj}): {round(F.item()/1000,2)} kN")
