import os
import sys
import numpy as np
import pandas as pd
import trackml.dataset

def select_hits(hits, truth, particles, pt_min=0):
    # Barrel volume and layer ids
    vlids = [(8,2), (8,4), (8,6), (8,8),
             (13,2), (13,4), (13,6), (13,8),
             (17,2), (17,4)]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])
    # Calculate particle transverse momentum
    pt = np.sqrt(particles.px**2 + particles.py**2)
    particles = particles.assign(pt=pt)
    # True particle selection.
    # Applies pt cut, removes all noise hits.
    particles = particles[particles.pt > pt_min]
    truth = (truth[['hit_id', 'particle_id']]
             .merge(particles[['particle_id', 'pt']], on='particle_id'))
    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    hits = (hits[['hit_id', 'z', 'layer']]
            .assign(r=r, phi=phi)
            .merge(truth[['hit_id', 'particle_id', 'pt']], on='hit_id'))
    # Remove duplicate hits
    hits = hits.loc[
        hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
    ]
    return hits

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def get_segments(hits, gid_keys, gid_start, gid_end):

    # Group hits by geometry ID
    hit_gid_groups = hits.groupby(gid_keys)

    segments = []

    # Loop over geometry ID pairs
    for gid1, gid2 in zip(gid_start, gid_end):
        hits1 = hit_gid_groups.get_group(gid1)
        hits2 = hit_gid_groups.get_group(gid2)

        # Join all hit pairs together
        hit_pairs = pd.merge(
            hits1.reset_index(), hits2.reset_index(),
            how='inner', on='evtid', suffixes=('_1', '_2'))

        # Calculate coordinate differences
        dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
        dz = hit_pairs.z_2 - hit_pairs.z_1
        dr = hit_pairs.r_2 - hit_pairs.r_1
        phi_slope = dphi / dr
        z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr

        # Identify the true pairs
        y = (hit_pairs.particle_id_1 == hit_pairs.particle_id_2) & (hit_pairs.particle_id_1 != 0)
        

        # Put the results in a new dataframe
        segments.append(hit_pairs[['evtid', 'index_1', 'index_2', 'layer_1', 'layer_2']]
                        .assign(dphi=dphi, dz=dz, dr=dr, y=y, phi_slope=phi_slope, z0=z0))

    return pd.concat(segments, ignore_index=True)

def select_segments(segments, phi_slope_min, phi_slope_max, z0_max):
    sel_mask = ((segments.phi_slope.abs() > phi_slope_min) &
                (segments.phi_slope.abs() < phi_slope_max) &
                (segments.z0.abs() < z0_max))
    return segments.assign(selected=sel_mask)

data_dir = 'data/train_100_events/'

n_events = 100

# Find the input files
all_files = os.listdir(data_dir)
suffix = '-hits.csv'
file_prefixes = sorted(os.path.join(data_dir, f.replace(suffix, ''))
                       for f in all_files if f.endswith(suffix))
file_prefixes = file_prefixes[:n_events]

hits_all = np.array([])
hits_1pT = np.array([])
true_segs_phi_slope = np.array([])
fake_segs_phi_slope = np.array([])
true_segs_z0 = np.array([])
fake_segs_z0 = np.array([])

gid_keys = 'layer'
n_det_layers = 10
gid_start = np.arange(0, n_det_layers-1)
gid_end = np.arange(1, n_det_layers)

# Choose some cuts
phi_slope_min = 0.
phi_slope_max = 0.0006
z0_max = 100

true_selected_segments = 0
true_segments = 0
all_selected_segments = 0
total_segments = 0

for prefix in file_prefixes:
    evtid = int(prefix[-9:])
    hits, particles, truth = trackml.dataset.load_event(
    prefix, parts=['hits', 'particles', 'truth'])

    # count total particles
    selected_hits = (select_hits(hits, truth, particles, pt_min=0.0)
        .assign(evtid=evtid)
        .reset_index(drop=True))
    hits_all = np.append(hits_all, selected_hits['pt'].to_numpy())

    # count > 1 GeV particles
    selected_hits = (select_hits(hits, truth, particles, pt_min=1.0)
        .assign(evtid=evtid)
        .reset_index(drop=True))
    hits_1pT = np.append(hits_1pT, selected_hits['pt'].to_numpy())

    # Obtain segments
    segments = get_segments(hits=selected_hits, gid_keys=gid_keys,
        gid_start=gid_start, gid_end=gid_end)

    true_segs = segments[segments.y]
    fake_segs = segments[segments.y == False]

    # Apply cuts to segments
    true_segs_phi_slope = np.append(true_segs_phi_slope,
        true_segs.phi_slope.to_numpy())
    fake_segs_phi_slope = np.append(fake_segs_phi_slope,
        fake_segs.phi_slope.to_numpy())
    true_segs_z0 = np.append(true_segs_z0,
        true_segs.z0.to_numpy())
    fake_segs_z0 = np.append(fake_segs_z0,
        fake_segs.z0.to_numpy())

    # Get selected segments
    segments = select_segments(segments, phi_slope_min=phi_slope_min,
                           phi_slope_max=phi_slope_max, z0_max=z0_max)

    true_selected_segments += (segments.y & segments.selected).sum()
    true_segments += segments.y.sum()
    all_selected_segments += segments.selected.sum()
    total_segments += len(segments)
    

print('All particles: ', len(hits_all))
print('> 1 GeV pT particles: ', len(hits_1pT))

print('Total segments: %d'%(total_segments))
print('Selected segments: %d'%(all_selected_segments))

# Effiency of true edge selection
print('Effificeny: %.4f'%(true_selected_segments/true_segments))
# How many of the selected edges are true edges
print('Purity: %.4f'%(true_selected_segments/all_selected_segments))

# Log files to csv files
np.savetxt('logs/mu200_1pT/graph_construction/particles_all_pT.csv',
    hits_all, delimiter=",")
np.savetxt('logs/mu200_1pT/graph_construction/particles_1pT_pT.csv',
    hits_1pT, delimiter=",")
np.savetxt('logs/mu200_1pT/graph_construction/true_segs_phi_slope.csv',
    true_segs_phi_slope, delimiter=",")
np.savetxt('logs/mu200_1pT/graph_construction/fake_segs_phi_slope.csv',
    fake_segs_phi_slope, delimiter=",")
np.savetxt('logs/mu200_1pT/graph_construction/true_segs_z0.csv',
    true_segs_z0, delimiter=",")
np.savetxt('logs/mu200_1pT/graph_construction/fake_segs_z0.csv',
    fake_segs_z0, delimiter=",")
