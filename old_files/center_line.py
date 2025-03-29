import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

N_markers = 20

# Load files
path_markers_pos = "/home/kefe11/RawDataset/Datensaetze_Examination1_bb/AF066/GeometryWD/Smooth_Centerline_Markers_LEFT_positions.npy"
m_pos = np.load(path_markers_pos)
path_markers_vecs = "/home/kefe11/RawDataset/Datensaetze_Examination1_bb/AF066/GeometryWD/Smooth_Centerline_Markers_LEFT_vecs.npy"
m_vecs = np.load(path_markers_vecs)
path_ref_pos = "/home/kefe11/RawDataset/Datensaetze_Examination1_bb/AF066/GeometryWD/refmarkers_LEFT_positions.npy"
ref_pos = np.load(path_ref_pos)
path_ref_vecs = "/home/kefe11/RawDataset/Datensaetze_Examination1_bb/AF066/GeometryWD/refmarkers_LEFT_vecs.npy"
ref_vecs = np.load(path_ref_vecs)

# Marker point
x_ref, y_ref, z_ref = -36.9319, 22.2388, 41.0437
vx_ref, vy_ref, vz_ref = 0.264585, 0.323682, 0.908419
min_dist, closer_m, intersection = 100, 0, None
for i in range(0, np.shape(m_pos)[0]):
    direction = m_pos[i] - m_pos[i-1]
    denominator = np.dot(np.array([vx_ref, vy_ref, vz_ref]), direction)
    if np.isclose(denominator, 0):
        continue
    t = np.dot(np.array([vx_ref, vy_ref, vz_ref]), np.array([x_ref, y_ref, z_ref]) - m_pos[i-1]) / denominator
    inter = m_pos[i-1] + t*direction
    x, y, z = inter[0], inter[1], inter[2]
    distance = ((x_ref-x)**2+(y_ref-y)**2+(z_ref-z)**2)**0.5
    is_between = np.dot(m_pos[i]-inter, inter - m_pos[i-1]) >= 0
    if distance < min_dist and is_between: 
        min_dist, closer_m, intersection = distance, i, inter  
x0, y0, z0 = intersection[0], intersection[1], intersection[2]

# Original point
xyz_mm = m_pos[closer_m-N_markers//2:closer_m,:]
xyz_mp = m_pos[closer_m+1:closer_m+N_markers//2+1,:]

# List 
t = 0.7 / np.linalg.norm(np.array([vx_ref, vy_ref, vz_ref]))
dx, dy, dz = t*vx_ref, t*vy_ref, t*vz_ref
xyz_m = [[x0 - dx, y0 - dy, z0 - dz]]
xyz_p = [[x0 + dx, y0 + dy, z0 + dz]]
xyz_mm_new = np.concatenate((xyz_mm[:-2], np.array(xyz_m)))
xyz_mp_new = np.concatenate((np.array(xyz_p), xyz_mp[2:]))
xyz_new = np.concatenate((xyz_mm_new, np.array([[x0, y0, z0]]) ,xyz_mp_new))

u_new = np.linspace(0, 1, 17)
tck, u_m = splprep([xyz_new[:, 0], xyz_new[:, 1], xyz_new[:, 2]], s=0) 
xyz = splev(u_new, tck)
xyz_new = np.array(xyz).T

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot(xyz_new[:, 0], xyz_new[:, 1], xyz_new[:, 2], 'gx-')
ax.scatter(x0, y0, z0, c='r', marker='x')
ax.scatter(x_ref, y_ref, z_ref, c='r', marker='o')
plt.savefig("center_line.png")
ax = fig.add_subplot(122)
ax.plot(xyz_new[:, 0], xyz_new[:, 2], 'gx-')
ax.scatter(x0, z0, c='r', marker='x')
ax.scatter([x_ref - dx, x_ref, x_ref+ dx], [z_ref - dz, z_ref, z_ref+ dz], c='r', marker='o')
plt.savefig("center_line.png")


for i in range(len(xyz_new)):
    print(f'<Item Version="0">\n<_BaseItem Version="0"/>\n<pos> {-xyz_new[i][0]} {-xyz_new[i][1]} {xyz_new[i][2]} 0 0 0</pos></Item>')