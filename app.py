import streamlit as st
import copy
import math
import numpy as np 
import matplotlib.pyplot as plt 

# Page setup
st.set_page_config(page_title="Test Streamlit App")

# Title

st.title("Truss Solver Input Panel")

# --- MATERIAL PROPERTIES ---
st.subheader("Material Properties")

E = st.number_input("Young's Modulus E (N/m²)", value=200e9, format="%.2e")
A = st.number_input("Cross-sectional Area A (m²)", value=0.005, format="%.6f")

xFac = st.number_input("Scale factor for plotted displacements", value=500)


# --- NODAL COORDINATES ---
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
    "Enter nodal coordinates (one row per line, comma-separated):",
    value="\n".join([f"{row[0]}, {row[1]}" for row in default_nodes],),height=250)

# Convert text → numpy array
nodes = np.array([
    [float(x) for x in line.split(",")]
    for line in nodes_text.splitlines()
    if line.strip() != ""
])


# --- MEMBERS ---
st.subheader("Members [node_i, node_j]")

default_members = np.array([
    [1,2],
    [2,3],
    [3,4],
    [4,5],
    [5,6],
    [6,7],
    [7,8],
    [1,8],
    [2,8],
    [3,7],
    [4,6],
    [3,8],
    [3,6]
])

members_text = st.text_area(
    "Enter member connections (node_i, node_j):",
    value="\n".join([f"{row[0]}, {row[1]}" for row in default_members]), height=350
)

members = np.array([
    [int(x) for x in line.split(",")]
    for line in members_text.splitlines()
    if line.strip() != ""
])


# --- SUPPORTS ---
st.subheader("Support Conditions")

restrainedDoF = st.text_input(
    "Restrained degrees of freedom, started at 1 (comma-separated):",
    value="1, 2, 10"
)

restrainedDoF = [int(x.strip()) for x in restrainedDoF.split(",")]


# --- LOADING ---
st.subheader("External Loads")

# Force vector for 16 DOF
forceVector = np.zeros((16, 1))

dof_input = st.text_input(
    "Enter DOFs to apply forces, by index, started at 0 (comma-separated):",
    value="3, 5, 7"
)

# Convert to list of integers
force_dofs = [int(x.strip()) for x in dof_input.split(",") if x.strip().isdigit()]


st.subheader("Forces at Degrees of Freedom")

for dof in force_dofs:
    force = st.number_input(f"Force at DOF {dof} (N)", value=0.0)
    forceVector[dof] = force



# --- OUTPUT PREVIEW ---
st.subheader("Input Summary")

st.write("**E:**", E)
st.write("**A:**", A)
st.write("**xFac:**", xFac)

st.write("### Restrained DOF")
st.write(restrainedDoF)


st.subheader("Structure Visualization")

# Assume `nodes` and `members` are already defined (from user inputs)

fig = plt.figure()
axes = fig.add_axes([0.1,0.1,3,3])
axes.set_aspect('equal', adjustable='box')

# --- Plot members ---
for mbr in members:  
    node_i = mbr[0]
    node_j = mbr[1]   
    
    ix = nodes[node_i-1,0]
    iy = nodes[node_i-1,1]
    jx = nodes[node_j-1,0]
    jy = nodes[node_j-1,1]
    
    axes.plot([ix,jx],[iy,jy],'b') # Member

# --- Plot nodes ---
# --- Plot nodes WITH LABELS ---
for i, node in enumerate(nodes):
    x, y = node
    axes.plot(x, y, 'bo')
    axes.text(x + 0.5, y + 0.2, str(i+1), fontsize=20, color="red")


axes.set_xlabel('Distance (m)')
axes.set_ylabel('Distance (m)')
axes.set_title('Structure to Analyse')
axes.grid()

# --- Show in Streamlit ---
st.pyplot(fig)


#------------------------------------------------------------------------------

#Define a function to calculate member orientation and length
def memberOrientation(memberNo):
    memberIndex = memberNo-1 #Index identifying member in array of members
    node_i = members[memberIndex][0] #Node number for node i of this member
    node_j = members[memberIndex][1] #Node number for node j of this member

    xi = nodes[node_i-1][0] #x-coord for node i
    yi = nodes[node_i-1][1] #y-coord for node i
    xj = nodes[node_j-1][0] #x-coord for node j
    yj = nodes[node_j-1][1] #y-coord for node j
    
    #Angle of member with respect to horizontal axis
    
    dx = xj-xi #x-component of vector along member
    dy = yj-yi #y-component of vector along member
    mag = math.sqrt(dx**2 + dy**2) #Magnitude of vector (length of member)
    memberVector = np.array([dx,dy])#Member represented as a vector    
    
    #Need to capture quadrant first then appropriate reference axis and offset angle
    if(dx>0 and dy==0):
        theta = 0
    elif(dx==0 and dy>0):
        theta = math.pi/2
    elif(dx<0 and dy==0):
        theta = math.pi
    elif(dx==0 and dy<0):
        theta = 3*math.pi/2
    elif(dx>0 and dy>0):
        # 0<theta<90
        refVector = np.array([1,0]) # Vector describing the positive x-axis  
        theta = math.acos(refVector.dot(memberVector)/(mag))#Standard formula for the angle between two vectors        
    elif(dx<0 and dy>0): 
        # 90<theta<180
        refVector = np.array([0,1]) # Vector describing the positive y-axis  
        theta = (math.pi/2) + math.acos(refVector.dot(memberVector)/(mag))#Standard formula for the angle between two vectors
    elif(dx<0 and dy<0):
        # 180<theta<270
        refVector = np.array([-1,0]) # Vector describing the negative x-axis  
        theta = math.pi + math.acos(refVector.dot(memberVector)/(mag))#Standard formula for the angle between two vectors
    else:
        # 270<theta<360
        refVector = np.array([0,-1]) # Vector describing the negative y-axis  
        theta = (3*math.pi/2) + math.acos(refVector.dot(memberVector)/(mag))#Standard formula for the angle between two vectors
    
    return [theta, mag]

    #-------------------------------------------------------------------------------

    #Calculate orientation and length for each member and store
orientations = np.array([])#Initialise an array to hold orientations
lengths = np.array([]) #Initialise an array to hold lengths
for n, mbr in enumerate(members):
    [angle, length] = memberOrientation(n+1)
    orientations = np.append(orientations,angle)
    lengths = np.append(lengths,length)  

    #-------------------------------------------------------------------------------

    #Define a function to calculate the global stiffness matrix of an axially loaded bar
def calculateKg(memberNo):
    """
    Calculate the global stiffness matrix for an axially loaded bar
    memberNo: The member number 
    """    
    theta = orientations[memberNo-1]
    mag = lengths[memberNo-1]
        
    c = math.cos(theta)
    s = math.sin(theta)
    
    K11 = (E*A/mag)*np.array([[c**2,c*s],[c*s,s**2]]) #Top left quadrant of global stiffness matrix
    K12 = (E*A/mag)*np.array([[-c**2,-c*s],[-c*s,-s**2]]) #Top right quadrant of global stiffness matrix   
    K21 = (E*A/mag)*np.array([[-c**2,-c*s],[-c*s,-s**2]]) #Bottom left quadrant of global stiffness matrix   
    K22 = (E*A/mag)*np.array([[c**2,c*s],[c*s,s**2]]) #Bottom right quadrant of global stiffness matrix          
    
    return [K11, K12, K21,K22]


#-------------------------------------------------------------------------------
#nDoF = np.amax(members)*2 #Total number of degrees of freedom in the problem
numberOfNodes = len(nodes)
nDoF = numberOfNodes * 2
Kp = np.zeros((nDoF, nDoF))
#Kp = np.zeros([nDoF,nDoF]) #Initialise the primary stiffness matrix

for n, mbr in enumerate(members):
#note that enumerate adds a counter to an iterable (n)

    #Calculate the quadrants of the global stiffness matrix for the member
    [K11, K12, K21,K22] = calculateKg(n+1)

    node_i = mbr[0] #Node number for node i of this member
    node_j = mbr[1] #Node number for node j of this member
    
    #Primary stiffness matrix indices associated with each node
    #i.e. node 1 occupies indices 0 and 1 (accessed in Python with [0:2])
    ia = 2*node_i-2 #index 0
    ib = 2*node_i-1 #index 1
    ja = 2*node_j-2 #index 2
    jb = 2*node_j-1 #index 3
    Kp[ia:ib+1,ia:ib+1] = Kp[ia:ib+1,ia:ib+1] + K11
    Kp[ia:ib+1,ja:jb+1] = Kp[ia:ib+1,ja:jb+1] + K12
    Kp[ja:jb+1,ia:ib+1] = Kp[ja:jb+1,ia:ib+1] + K21
    Kp[ja:jb+1,ja:jb+1] = Kp[ja:jb+1,ja:jb+1] + K22


    #-------------------------------------------------------------------------------

    restrainedIndex = [x - 1 for x in restrainedDoF] #Index for each restrained DoF (list comprehension)

#Reduce to structure stiffness matrix by deleting rows and columns for restrained DoF
Ks = np.delete(Kp,restrainedIndex,0) #Delete rows
Ks = np.delete(Ks,restrainedIndex,1) #Delete columns
Ks = np.matrix(Ks) # Convert Ks from numpy.ndarray to numpy.matrix to use build in inverter function


    #-------------------------------------------------------------------------------


forceVectorRed = copy.copy(forceVector)# Make a copy of forceVector so the copy can be edited, leaving the original unchanged
forceVectorRed = np.delete(forceVectorRed,restrainedIndex,0) #Delete rows corresponding to restrained DoF
U = np.linalg.solve(Ks, forceVectorRed)
#U = Ks.I*forceVectorRed 




    #-------------------------------------------------------------------------------

#Construct the global displacement vector
UG = np.zeros(nDoF) #Initialise an array to hold the global displacement vector
c=0 #Initialise a counter to track how many restraints have been imposed
for i in np.arange(nDoF):    
    if i in restrainedIndex:
        #Impose zero displacement
        UG[i] = 0        
    else:
        #Assign actual displacement
        UG[i] = U[c, 0]
        c=c+1

UG = np.array([UG]).T  
FG = np.matmul(Kp,UG)

#Generate output statements
for i in np.arange(0,len(restrainedIndex)):           
    index = restrainedIndex[i]


 #-------------------------------------------------------------------------------

mbrForces = np.array([]) #Initialise an array to hold member forces
for n, mbr in enumerate(members):    
    theta = orientations[n]
    mag = lengths[n]
    
    node_i = mbr[0] #Node number for node i of this member
    node_j = mbr[1] #Node number for node j of this member    
    #Primary stiffness matrix indices associated with each node
    ia = 2*node_i-2 
    ib = 2*node_i-1 
    ja = 2*node_j-2 
    jb = 2*node_j-1 
    
    #Transformation matrix
    c = math.cos(theta)
    s = math.sin(theta)
    T = np.array([[c,s,0,0],[0,0,c,s]])
    
    disp = np.array([[UG[ia],UG[ib],UG[ja],UG[jb]]]).T #Glocal displacements
    disp_local = np.matmul(T,disp)[0] #Local displacements    
    F_axial = (A*E/mag)*(disp_local[1]-disp_local[0]) #Axial loads    
    mbrForces = np.append(mbrForces,F_axial) #Store axial loads


 #-------------------------------------------------------------------------------

    #FIGURE TO PLOT TENSION/COMPRESSION MEMBERS
fig = plt.figure() 
axes = fig.add_axes([0.1,0.1,2,2]) 
fig.gca().set_aspect('equal', adjustable='box')

#Plot members
for n, mbr in enumerate(members):  
    node_i = mbr[0] #Node number for node i of this member
    node_j = mbr[1] #Node number for node j of this member   
    
    ix = nodes[node_i-1,0] #x-coord of node i of this member
    iy = nodes[node_i-1,1] #y-coord of node i of this member
    jx = nodes[node_j-1,0] #x-coord of node j of this member
    jy = nodes[node_j-1,1] #y-coord of node j of this member
    
    #Index of DoF for this member
    ia = 2*node_i-2 #horizontal DoF at node i of this member 
    ib = 2*node_i-1 #vertical DoF at node i of this member
    ja = 2*node_j-2 #horizontal DoF at node j of this member
    jb = 2*node_j-1 #vertical DoF at node j of this member
    
    if(abs(mbrForces[n])<0.001):
        axes.plot([ix,jx],[iy,jy],'grey',linestyle='--') #Zero force in member
    elif(mbrForces[n]>0):
        axes.plot([ix,jx],[iy,jy],'b') #Member in tension
    else:
        axes.plot([ix,jx],[iy,jy],'r') #Member in compression
    
#Plot nodes
for node in nodes:
    axes.plot([node[0]],[node[1]],'bo', ms=3) 

axes.set_xlabel('Distance (m)')
axes.set_ylabel('Distance (m)')
axes.set_title('Tension/compression members')
axes.grid()
st.pyplot(fig)

#=========================================================================

#FIGURE TO PLOT DEFLECTED SHAPE
fig = plt.figure() 
axes = fig.add_axes([0.1,0.1,2,2]) 
fig.gca().set_aspect('equal', adjustable='box')

#Plot members
for mbr in members:  
    node_i = mbr[0] #Node number for node i of this member
    node_j = mbr[1] #Node number for node j of this member   
    
    ix = nodes[node_i-1,0] #x-coord of node i of this member
    iy = nodes[node_i-1,1] #y-coord of node i of this member
    jx = nodes[node_j-1,0] #x-coord of node j of this member
    jy = nodes[node_j-1,1] #y-coord of node j of this member
    
    #Index of DoF for this member
    ia = 2*node_i-2 #horizontal DoF at node i of this member 
    ib = 2*node_i-1 #vertical DoF at node i of this member
    ja = 2*node_j-2 #horizontal DoF at node j of this member
    jb = 2*node_j-1 #vertical DoF at node j of this member
    
    axes.plot([ix,jx],[iy,jy],'grey', lw=0.75) #Member
    axes.plot([ix + UG[ia,0]*xFac, jx + UG[ja,0]*xFac], [iy + UG[ib,0]*xFac, jy + UG[jb,0]*xFac],'r') #Deformed member
    
axes.set_xlabel('Distance (m)')
axes.set_ylabel('Distance (m)')
axes.set_title('Deflected shape')
axes.grid()
st.pyplot(fig)

#--------------------------------------------------------------------

#Generate output statements
import streamlit as st

# --- REACTIONS ---
st.subheader("Reactions")
for i in np.arange(0, len(restrainedIndex)):
    index = restrainedIndex[i]
    st.write(
        f"Reaction at DoF **{index+1}**: **{round(FG[index].item()/1000, 2)} kN**"
    )

# --- MEMBER FORCES ---
st.subheader("Member Forces")
for n, mbr in enumerate(members):
    st.write(
        f"Force in member **{n+1}** (nodes {mbr[0]} to {mbr[1]}) = **{round(mbrForces[n]/1000, 2)} kN**"
    )

# --- NODAL DISPLACEMENTS ---
st.subheader("Nodal Displacements")
for n, node in enumerate(nodes):
    ix = 2*(n+1)-2
    iy = 2*(n+1)-1

    ux = round(UG[ix, 0], 5)
    uy = round(UG[iy, 0], 5)

    st.write(
        f"Node **{n+1}**: Ux = **{ux} m**, Uy = **{uy} m**"
    )
