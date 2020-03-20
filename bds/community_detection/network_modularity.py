"""
Analysis of brain network modularity in depressed and non-depressed
individuals based on ECoG correlational structure.

Author: Ankit N Khambhati <akhambhati@gmail.com>

Last Updated: 2018/11/07
"""

import glob
import os
import re
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import modularity as mdlr
import numpy as np
import pandas as pd
import scipy
import scipy.io as io
import scipy.stats as sp_stats
import seaborn as sns
import networkx as nx

import ncat
import nibabel as nib
import supereeg as se
import xmltodict

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Define paths where relevant data is held
DIR_AAL2 = '/home/kscangos/Sandbox/data/AAL2'
DIR_YEO = '/home/kscangos/Sandbox/data/Yeo_JNeurophysiol11_MNI152/'
DIR_LAUSANNE = '/home/kscangos/Sandbox/data/Lausanne/'
DIR_GORDON = '/home/kscangos/Sandbox/data/Gordon2016/'
#DIR_MODEL = '/userdata/pdaly/supereeg/models/samp100_10h_54pid'
DIR_MODEL = '/userdata/pdaly/supereeg/models/samp100_2h_54pid'
#DIR_MODEL = '/userdata/pdaly/supereeg/models/samp80_2h_60pid'
#DIR_MODEL = '/userdata/pdaly/supereeg/models/one_sampled'
#DIR_MODEL = '/userdata/kscangos/Subnets/Signals/SuperEEG/Sandbox/results/models/ANK'
DIR_SUPEREEG = '/userdata/kscangos/Subnets/Signals/SuperEEG'
#DIR_BO = '/userdata/kscangos/Subnets/Signals/SuperEEG/brainobjects'
DIR_BO = '/userdata/kscangos/Subnets/Signals/SuperEEG/brainobjects_sampled_all_2h_fixed_newpts'
DIR_MODULARITY = '/userdata/kscangos/Subnets/Signals/SuperEEG/modularity'
DF_DEPR = pd.read_table('/home/kscangos/Sandbox/full_patient_list_ankit')
DF_FULL = pd.read_csv(
    '/home/kscangos/Sandbox/full_patient_list_pd_feb.csv', index_col=0)


def _partition_model(model_path, gamma, cache_path):
    model = se.Model(model_path)
    adj_matr = np.abs(model.get_model())
    cfg_matr = mdlr.matrix.convert_adj_matr_to_conn_vec(adj_matr).reshape(
        1, -1)

    B, twom = mdlr.matrix.super_modularity_matr(
        cfg_matr, gamma=gamma, omega=0.0)
    partition, Q = mdlr.genlouvain(B, limit=1000, verbose=False)

    Q /= twom

    with open(cache_path, 'w') as file:
        np.savez(file, partition=partition, Q=Q, gamma=gamma)


def _wrap_partition_model(kwargs):
    return _partition_model(**kwargs)


def map_modularity(model_base_name, n_run, n_pool=5, gamma_list=[1.12]):
    # 1st
    # TODO: changed gamma_list to a single value, may need to adjust!
    model_path_list = glob.glob('{}/{}*.mo'.format(DIR_MODEL, model_base_name))

    # Create run list
    run_list = []
    for path in model_path_list:
        model_name = path.split('/')[-1].split('.')[0]

        # Get last known ID used to cache results
        model_cache_id = [
            int(pth.split('/')[-1].split('.')[-2]) for pth in glob.glob(
                '{}/{}.*.mdlr'.format(DIR_MODULARITY, model_name))
        ]
        start_id = 0 if len(model_cache_id) == 0 else (
            np.max(model_cache_id) + 1)

        for gamma in gamma_list:
            for run in range(n_run):

                cache_path = '{}/{}.{}.mdlr'.format(DIR_MODULARITY, model_name,
                                                    start_id)

                run_list.append({
                    'model_path': path,
                    'gamma': gamma,
                    'cache_path': cache_path
                })

                start_id += 1

    # Spawn pool
    n_pool = 1 if n_pool <= 1 else n_pool
    pp = Pool(n_pool)
    pp.map(_wrap_partition_model, run_list)


def reduce_Q(model_base_name):
    modularity_list = glob.glob('{}/{}*.mdlr'.format(DIR_MODULARITY,
                                                     model_base_name))

    df_dict = {'Sub': [], 'gamma': [], 'Q': []}

    for path in modularity_list:
        df = np.load(path)

        pt_num = path.split('/')[-1].split('.')[0]
        if 'EC{}'.format(pt_num) in np.array(DF_DEPR['Sub']):
            pt_id = 'EC{}'.format(pt_num)
        elif 'KP{}'.format(pt_num) in np.array(DF_DEPR['Sub']):
            pt_id = 'KP{}'.format(pt_num)
        else:
            pt_id = pt_num
        df_dict['Sub'].append(pt_id)
        df_dict['gamma'].append(df['gamma'][()])
        df_dict['Q'].append(df['Q'][()])

    df = pd.DataFrame.from_dict(df_dict)
    return df
    # Plot
    df_merge = pd.merge(DF_DEPR, df, on='Sub')
    df_merge_subj = df_merge.groupby(['Sub', 'gamma']).mean().reset_index()
    df_merge_subj = df_merge_subj[df_merge_subj['Sub'] != 'EC154']
    df_merge_subj = df_merge_subj[df_merge_subj['Sub'] != 'KP20']
    df_merge_subj = df_merge_subj[df_merge_subj['Sub'] != 'EC111']
    plt.figure(figsize=(24, 6), dpi=300)
    ax = plt.subplot(111)
    ax = sns.lineplot(x='gamma', y='Q', hue='Dep', ci=68, data=df_merge_subj)
    ax.set_xlabel('Module Resolution (gamma)')
    ax.set_ylabel('Modularity (Q)')

    # Run a functional curve analysis
    df_sel = df_merge_subj.groupby(['Dep', 'gamma']).mean().reset_index()
    df_sel = df_sel[df_sel['gamma'] <= 2.0]
    true_diff = np.sum((df_sel[df_sel['Dep'] == 0].reset_index() -
                        df_sel[df_sel['Dep'] == 1].reset_index()))['Q']

    n_perm = 10000
    null_diff = []
    DF_DEPR_NULL = DF_DEPR.copy()
    for perm_i in range(n_perm):
        DF_DEPR_NULL['Dep'] = np.random.permutation(DF_DEPR_NULL['Dep'])
        df_merge = pd.merge(DF_DEPR_NULL, df, on='Sub')
        df_merge_subj = df_merge.groupby(['Sub', 'gamma']).mean().reset_index()
        df_merge_subj = df_merge_subj[df_merge_subj['Sub'] != 'EC154']
        df_merge_subj = df_merge_subj[df_merge_subj['Sub'] != 'KP20']
        df_merge_subj = df_merge_subj[df_merge_subj['Sub'] != 'EC111']
        df_sel = df_merge_subj.groupby(['Dep', 'gamma']).mean().reset_index()
        df_sel = df_sel[df_sel['gamma'] <= 2.0]
        null_diff.append(
            np.sum((df_sel[df_sel['Dep'] == 0].reset_index() -
                    df_sel[df_sel['Dep'] == 1].reset_index()))['Q'])
    null_diff = np.array(null_diff)

    ax.set_title('p={}'.format(np.mean(null_diff > true_diff)))
    plt.show()

    return df


def calc_Q(adj_matr, comm_vec, gamma):
    cfg_matr = mdlr.matrix.convert_adj_matr_to_conn_vec(adj_matr).reshape(
        1, -1)
    B, twom = mdlr.matrix.super_modularity_matr(
        cfg_matr, gamma=gamma, omega=0.0)
    Q = np.sum(np.diag(mdlr._metanetwork(B, comm_vec)))
    return Q


def reduce_Q_held_out(model_base_name):
    loo_list = glob.glob('{}_loos/*.mo'.format(DIR_MODEL))
    df_cons = np.load('{}/{}_full-pos.cons'.format(DIR_MODULARITY,
                                                   model_base_name))
    mo_full = se.load('{}/{}_full-pos.mo'.format(DIR_MODEL,
                                                 model_base_name)).get_model()

    Q_dict = {'pid': [], 'gamma': [], 'Qfull-Qheld': [], 'Qheld': []}
    Q_cache = {}
    for loo_pth in loo_list:
        pid = loo_pth.split('/')[-1].split('-')[0].split('_')[-1]
        print(pid)
        mo_loo = se.load(loo_pth).get_model()

        for gamma in df_cons.keys():
            massign = df_cons[gamma]

            if gamma not in Q_cache:
                Q_cache[gamma] = calc_Q(mo_full, massign, float(gamma))
            Qheld = calc_Q(mo_loo, massign, float(gamma))

            Q_dict['pid'].append(pid)
            Q_dict['gamma'].append(float(gamma))
            Q_dict['Qfull-Qheld'].append(Q_cache[gamma] - Qheld)
            Q_dict['Qheld'].append(Qheld)

    Q_dict = pd.DataFrame.from_dict(Q_dict)

    Q_dict.to_csv('{}/{}_full-pos.Qfull_Qheld.csv'.format(
        DIR_MODULARITY, model_base_name))

    return Q_dict


def reduce_allegiance(model_base_name):
    # 2nd
    modularity_list = glob.glob('{}/{}*.mdlr'.format(DIR_MODULARITY,
                                                     model_base_name))

    model_name_list = np.unique(
        [path.split('/')[-1].split('.')[0] for path in modularity_list])

    for name in model_name_list:

        cache_path = '{}/{}.allg'.format(DIR_MODULARITY, name)
        if os.path.exists(cache_path):
            print('Already built allegiance matrices: {}'.format(name))
            continue
        print('Building allegiance matrices: {}'.format(name))

        modularity_name_list = glob.glob('{}/{}.*.mdlr'.format(
            DIR_MODULARITY, name))

        df_dict = {}
        for path in modularity_name_list:
            print(path)
            df = np.load(path)

            partition = df['partition']
            gamma = df['gamma'][()]
            gamma = str(gamma)

            if gamma not in df_dict:
                df_dict[gamma] = {
                    'allegiance': np.zeros((len(partition), len(partition))),
                    'count': 0
                }

            for m_id in np.unique(partition):
                in_part = partition == m_id
                df_dict[gamma]['allegiance'] += \
                    in_part.reshape(-1, 1).dot(in_part.reshape(1, -1))
            df_dict[gamma]['count'] += 1

        with open(cache_path, 'w') as file:
            np.savez(file, **df_dict)


def reduce_consensus(model_base_name):
    # 3rd
    allegiance_list = glob.glob('{}/{}*.allg'.format(DIR_MODULARITY,
                                                     model_base_name))

    df_dict = {}
    for path in allegiance_list:
        name = path.split('/')[-1].split('.')[0]

        cache_path = '{}/{}.cons'.format(DIR_MODULARITY, name)
        if os.path.exists(cache_path):
            print('Already built consensus partitions: {}'.format(name))
            continue
        print('Building consensus partitions: {}'.format(name))

        df = np.load(path)

        for gamma in df.keys():
            df_g = df[gamma][()]
            adj_matr = df_g['allegiance'] / df_g['count']
            cfg_matr = mdlr.matrix.convert_adj_matr_to_conn_vec(
                adj_matr).reshape(1, -1)
            B, twom = mdlr.matrix.super_modularity_matr(
                cfg_matr, gamma=1.0, omega=0.0)
            partition, Q = mdlr.genlouvain(B, limit=1000, verbose=False)

            df_dict[gamma] = partition

        with open(cache_path, 'w') as file:
            np.savez(file, **df_dict)


def plot_ordered_matrices(model_name):
    full_corr_mat = se.Model('{}/{}.mo'.format(DIR_MODEL,
                                               model_name)).get_model()
    full_corr_mat[np.diag_indices_from(full_corr_mat)] = 0
    full_corr_mat = np.abs(full_corr_mat)

    df_cons = np.load('{}/{}.cons'.format(DIR_MODULARITY, model_name))

    df_allg = np.load('{}/{}.allg'.format(DIR_MODULARITY, model_name))

    for gamma in [
            '0.9583333333333333', '1.1875', '2.1041666666666665',
            '3.020833333333333'
    ]:

        # df_cons.keys():

        massign = df_cons[gamma]
        massign_ix = np.argsort(massign)
        m_id, m_ix = np.unique(massign[massign_ix], return_index=True)
        allg = df_allg[gamma][()]['allegiance'] / df_allg[gamma][()]['count']

        plt.figure(figsize=(4, 4), dpi=300)
        ax = plt.subplot(111)
        mat = ax.matshow(
            full_corr_mat[massign_ix, :][:, massign_ix],
            vmin=0.2,
            vmax=0.4,
            cmap='inferno')
        cb = plt.colorbar(mat, ax=ax)
        cb.set_label('Functional Connectivity', rotation=270)
        """
        ax.vlines(
            m_ix - 0.5,
            -0.5,
            len(full_corr_mat) - 0.5,
            color='w',
            linewidth=1.0)
        ax.hlines(
            m_ix - 0.5,
            -0.5,
            len(full_corr_mat) - 0.5,
            color='w',
            linewidth=1.0)
        """
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(m_ix)
        ax.set_xticklabels([])
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks(m_ix)
        ax.set_yticklabels([])
        ax.set_xlabel('Electrodes (nodes)')
        ax.set_ylabel('Electrodes (nodes)')
        ax.set_title('Gamma = {:0.2f}'.format(float(gamma)))
        plt.savefig(
            './network_modularity_figs/FC_Matr.{:0.2f}.{}.pdf'.format(
                float(gamma), model_name),
            transparent=True)
        plt.close()

        plt.figure(figsize=(4, 4), dpi=300)
        ax = plt.subplot(111)
        mat = ax.matshow(
            allg[massign_ix, :][:, massign_ix],
            vmin=0.0,
            vmax=1.0,
            cmap='viridis')
        cb = plt.colorbar(mat, ax=ax, ticks=[0.0, 0.5, 1.0])
        cb.set_label('Module Allegiance', rotation=270)
        """
        ax.vlines(
            m_ix - 0.5,
            -0.5,
            len(full_corr_mat) - 0.5,
            color='w',
            linewidth=1.0)
        ax.hlines(
            m_ix - 0.5,
            -0.5,
            len(full_corr_mat) - 0.5,
            color='w',
            linewidth=1.0)
        """
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticklabels([])
        ax.yaxis.set_ticks_position('left')
        ax.set_yticklabels([])
        ax.set_xlabel('Electrodes (nodes)')
        ax.set_ylabel('Electrodes (nodes)')
        ax.set_title('Gamma = {:0.2f}'.format(float(gamma)))
        plt.savefig(
            './network_modularity_figs/ModAlleg.{:0.2f}.{}.pdf'.format(
                float(gamma), model_name),
            transparent=True)
        plt.close()


def tabulate_consensus(model_base_name):
    # 4rd
    consensus_list = glob.glob('{}/{}*.cons'.format(DIR_MODULARITY,
                                                    model_base_name))

    for path in consensus_list:
        name = path.split('/')[-1]

        cache_path = '{}/{}.csv'.format(DIR_MODULARITY, name)
        if os.path.exists(cache_path):
            print('Already tabulated consensus partitions: {}'.format(name))
            continue
        print('Tabulating consensus partitions: {}'.format(name))

        df = np.load(path)

        tab_dict = {'electrode': [], 'gamma': [], 'cluster_id': []}

        for gamma in df.keys():
            df_g = df[gamma][()]
            gamma = float(gamma)
            for e_ii, c_id in enumerate(df_g):
                tab_dict['gamma'].append(gamma)
                tab_dict['electrode'].append(e_ii)
                tab_dict['cluster_id'].append(c_id)

        df = pd.DataFrame(tab_dict)
        df.to_csv(cache_path, index=False)


def tabulate_atlas(model_base_name):
    full_corr, dep_corr, nodep_corr, massign, locs = load_all_models(
        model_base_name)
    yeo_name = assign_locs_to_yeo_roi(np.array(locs))
    yeosubg_name = assign_locs_to_yeosubg_roi(np.array(locs))
    laus_name = assign_locs_to_lausanne_roi(np.array(locs))
    gordon_name = assign_locs_to_gordon_roi(np.array(locs))
    cache_path = '{}/{}.atlas.csv'.format(DIR_MODULARITY, model_base_name)

    tab_dict = {
        'electrode': [],
        'Yeo_Label': [],
        'YeoSubg_Label': [],
        'Lausanne_Label': [],
        'Gordon_Label': []
    }
    for e_ii, (yname, ysname, lname, gname) in enumerate(
            zip(yeo_name, yeosubg_name, laus_name, gordon_name)):
        tab_dict['electrode'].append(e_ii)
        tab_dict['Yeo_Label'].append(yname)
        tab_dict['YeoSubg_Label'].append(ysname)
        tab_dict['Lausanne_Label'].append(lname)
        tab_dict['Gordon_Label'].append(gname)


    df = pd.DataFrame(tab_dict)
    df.to_csv(cache_path, index=False)


def kl_divergence(P, Q):
    assert len(P) == len(Q)

    P = P.copy()
    Q = Q.copy()
    P /= np.sum(P)
    Q /= np.sum(Q)

    return -1 * np.sum(P * np.log10(Q / P))


def subsample_electrodes(elec_lbl, frac_remove):
    assert (frac_remove >= 0) & (frac_remove <= 1)

    # Get the unique electrode labels
    lbl_id, lbl_cnt = np.unique(elec_lbl, return_counts=True)
    lbl_cnt = np.array(lbl_cnt, dtype=float)

    lbl_id = lbl_id[np.argsort(lbl_cnt)]
    lbl_cnt = lbl_cnt[np.argsort(lbl_cnt)]

    n_elec = len(elec_lbl)
    n_lbl = len(lbl_id)

    # Create reference counts
    Q = (n_elec / n_lbl) * np.ones(n_lbl)

    # Create true counts
    P = lbl_cnt.copy()

    # Randomly remove electrodes from the valid selection indices
    # to minmize the KL divergence
    KL0 = kl_divergence(P, Q)
    i = 0
    rm_inds = []
    print('Initial KL-Divergence: {:0.3f}'.format(KL0))
    while (len(rm_inds) < (frac_remove * n_elec)):
        i += 1

        PP = P.copy()
        rm_lbl_ix = np.random.permutation(n_lbl)[0]
        PP[rm_lbl_ix] -= 1

        KL = kl_divergence(PP, Q)
        if KL < KL0:
            P = PP
            KL0 = KL
            rm_inds.append(
                np.random.permutation(
                    np.flatnonzero(elec_lbl == lbl_id[rm_lbl_ix]))[0])
            i = 0

            print('Total electrodes removed: {}  | '.format(len(rm_inds)) +
                  'KL-Divergence: {:0.3f}'.format(KL0))

        if i == 2 * n_elec:
            break

    rm_inds = np.array(rm_inds)
    keep_inds = np.setdiff1d(np.arange(n_elec), rm_inds)
    return keep_inds, rm_inds


def distrib_remove_elec(frac_rm=0.25):
    bo_list = glob.glob('{}/*.bo'.format(DIR_BO))
    pid_list = np.unique(
        [bo_name.split('/')[-1].split('_')[0] for bo_name in bo_list])
    pid_list = np.array(['pid192'])

    # Iterate over subjects
    pt_rm_dict = {'PID': [], 'ROI': [], 'N_ELEC': [], 'TYPE': []}
    for pid in pid_list:
        print(pid)

        # Get the full set of locations across BOs
        pid_bo_list = glob.glob('{}/{}_*.bo'.format(DIR_BO, pid))

        pid_locs = np.empty((0, 3))
        for pid_path in pid_bo_list:
            bo = se.load(pid_path)
            pid_locs = np.concatenate(
                (pid_locs, np.array(bo.get_locs())), axis=0)
        pid_locs = np.unique(pid_locs, axis=0)

        # Assign the locations to Lausanne atlas ROIs
        pid_roi = assign_locs_to_lausanne_roi(pid_locs)

        # Find electrodes to remove
        kp_ind, rm_ind = subsample_electrodes(pid_roi, frac_remove=frac_rm)

        # Get stats on which ROIs are removed
        roi_lbl, roi_cnt = np.unique(pid_roi, return_counts=True)
        for rtype in ['Before', 'After']:
            for ll, cc in zip(roi_lbl, roi_cnt):
                pt_rm_dict['PID'].append(pid)
                pt_rm_dict['ROI'].append(ll)
                if rtype == 'Before':
                    pt_rm_dict['N_ELEC'].append(cc)
                else:
                    pt_rm_dict['N_ELEC'].append((pid_roi[kp_ind] == ll).sum())
                pt_rm_dict['TYPE'].append(rtype)

        # Save the subsampled CSV
        pd.DataFrame(pid_locs[kp_ind]).to_csv(
            '{}/subsampled_locs/{}_2h-all_sampled-frac_rm_{}.csv'.format(
                DIR_SUPEREEG, pid, frac_rm),
            index=False,
            header=False)
    """
    pt_rm_dict = pd.DataFrame.from_dict(pt_rm_dict)
    pt_rm_dict.to_csv(
        '{}/subsampled_locs/ROI_Distribution-first_file-frac_rm_{}.csv'.format(
            DIR_SUPEREEG, frac_rm))
    """


def subsample_grid(eleclabels, subsample_spacing):

    # Find all electrodes of grid subtype
    grid_ix = np.flatnonzero(eleclabels[:, 2] == 'grid')
    nongrid_ix = np.flatnonzero(eleclabels[:, 2] != 'grid')

    if len(grid_ix) == 0:
        return nongrid_ix

    # Select grid electrodes
    sel_eleclabels = eleclabels[grid_ix, :]

    # Get grid names
    grid_name = np.array(
        [re.sub(r'\d+', '', name) for name in sel_eleclabels[:, 1]])

    # Iterate over grid names
    subsample_ix = []
    for name in np.unique(grid_name):
        name_ix = np.flatnonzero(grid_name == name)
        name_ix = name_ix[::subsample_spacing]
        for ix in name_ix:
            subsample_ix.append(ix)
    subsample_ix = np.array(subsample_ix)

    return np.union1d(nongrid_ix, grid_ix[subsample_ix])


def distrib_remove_grid():
    subspace = 2
    pid_list = glob.glob(
        '{}/*/clinical_elecs_all_warped.csv'.format(DIR_SUPEREEG))

    # Individual Analysis
    pt_rm_dict = {'PID': [], 'ROI': [], 'N_Elecs_Removed': []}
    for pid_i, csv_path in enumerate(pid_list):
        print('{} of {}'.format(pid_i + 1, len(pid_list)))

        pid = csv_path.split('/')[-2]

        # Get the electrode MAT file for the subject
        elec_mat_path1 = glob.glob(
            '{}/{}/elecs/clinical_TDT_elecs_all_warped.mat'.format(
                '/data_store2/imaging/subjects', pid))
        elec_mat_path2 = glob.glob(
            '{}/{}/elecs/clinical_elecs_all_warped.mat'.format(
                '/data_store2/imaging/subjects', pid))

        try:
            elec_mat = io.loadmat(elec_mat_path1[0], squeeze_me=True)
            e_coord = elec_mat['elecmatrix']
            val = np.flatnonzero(~np.isnan(np.sum(e_coord, axis=1)))
            e_coord = e_coord[val]
            if 'anatomy' in elec_mat:
                e_label = elec_mat['anatomy'][val, :]
            elif 'eleclabels' in elec_mat:
                e_label = elec_mat['eleclabels'][val, :]
            sel_ix = subsample_grid(e_label, 2)
            e_coord = e_coord[sel_ix, :]
        except:
            try:
                elec_mat = io.loadmat(elec_mat_path2[0], squeeze_me=True)
                e_coord = elec_mat['elecmatrix']
                val = np.flatnonzero(~np.isnan(np.sum(e_coord, axis=1)))
                e_coord = e_coord[val]
                if 'anatomy' in elec_mat:
                    e_label = elec_mat['anatomy'][val, :]
                elif 'eleclabels' in elec_mat:
                    e_label = elec_mat['eleclabels'][val, :]
                sel_ix = subsample_grid(e_label, 2)
                e_coord = e_coord[sel_ix, :]
            except Exception as E:
                print('Skipping: {}'.format(pid))
                print('        {}'.format(E))
                continue

        # Get the electrode CSV file for the subject
        elec_csv = np.array(pd.read_csv(csv_path, header=None))
        csv_roi = assign_locs_to_lausanne_roi(elec_csv)

        # Keep csv locs contained in filtered elec_mat
        loc_ix = []
        for ix, csv_crd in enumerate(elec_csv):
            find = np.sum((e_coord - csv_crd)**2, axis=1)
            if np.min(find) < 1e-4:
                loc_ix.append(np.argmin(find))
        loc_ix = np.array(loc_ix)
        if len(loc_ix) == 0:
            print('Skipping: {}'.format(pid))
            print('        No matches between BO and elec.mat')
            continue

        csv_roi_filter = assign_locs_to_lausanne_roi(elec_csv[loc_ix])

        # Save the subsampled CSV
        pd.DataFrame(elec_csv).to_csv(
            '{}/{}/clinical_elecs_all_warped-subsample_grid_{}.csv'.format(
                DIR_SUPEREEG, pid, subspace),
            index=False,
            header=False)

        # Append to dataframe to assess which ROIs lost most electrodes
        for roi_name in np.unique(csv_roi):
            n_roi_rm = np.sum(csv_roi == roi_name) - np.sum(
                csv_roi_filter == roi_name)

            pt_rm_dict['PID'].append(pid)
            pt_rm_dict['ROI'].append(roi_name)
            pt_rm_dict['N_Elecs_Removed'].append(n_roi_rm)

    # Individual Plot
    pt_rm_dict = pd.DataFrame.from_dict(pt_rm_dict)
    roi_order = np.array(
        pt_rm_dict.groupby([
            'ROI'
        ]).mean().reset_index().sort_values(by='N_Elecs_Removed')['ROI'])[::-1]
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    ax = sns.boxplot(
        x='ROI', y='N_Elecs_Removed', order=roi_order, data=pt_rm_dict)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title('Sub-sample grid electrodes by factor {}'.format(subspace))
    plt.tight_layout()

    plt.show()


def assign_locs_to_aal_roi(elec_locs):
    nifti = nib.load('{}/aal2.nii.gz'.format(DIR_AAL2))
    nifti_lbl = np.loadtxt('{}/aal2.nii.txt'.format(DIR_AAL2), dtype=str)
    nifti_id = np.array(nifti_lbl[:, 0], dtype=float)
    nifti_lbl = nifti_lbl[:, 1]
    nifti_img = nifti.get_data()
    nifti_img[nifti_img > 94] = 0  # Block out cerebellum and vermis

    lbl_xyz = {}
    for lbl, lbl_id in zip(nifti_lbl, nifti_id):
        lbl_ijk = np.array(np.nonzero(nifti_img == lbl_id)).T
        if lbl_ijk.shape[0] == 0:
            continue
        lbl_ijk = np.concatenate((lbl_ijk, np.ones((len(lbl_ijk), 1))), axis=1)
        lbl_xyz[lbl] = nifti.affine.dot(lbl_ijk.T)[:3, :].T

    elec_lbls = []
    for xyz in elec_locs:
        min_dists = np.array([
            np.nanmin(np.sqrt(np.nansum((lbl_xyz[lbl] - xyz)**2, axis=1)))
            for lbl in lbl_xyz
        ])
        elec_lbls.append(lbl_xyz.keys()[np.argmin(min_dists)])
    elec_lbls = np.array(elec_lbls)

    return elec_lbls


def assign_locs_to_gordon_roi(elec_locs):
    nifti = nib.load('{}/Parcels_MNI_111.nii'.format(DIR_GORDON))
    nifti_lbl = pd.read_excel('{}/Parcels.xlsx'.format(DIR_GORDON))
    nifti_id = np.array(nifti_lbl['ParcelID'], dtype=float)
    nifti_lbl = np.array(['_'.join(i[1][['Community', 'Hem']]) \
        for i in nifti_lbl.iterrows()])
    nifti_img = nifti.get_data()[:, :, :]

    lbl_xyz = {}
    for lbl, lbl_id in zip(nifti_lbl, nifti_id):
        lbl_ijk = np.array(np.nonzero(nifti_img == lbl_id)).T
        if lbl_ijk.shape[0] == 0:
            print('{}, {} not found.'.format(lbl, lbl_id))
            continue
        lbl_ijk = np.concatenate((lbl_ijk, np.ones((len(lbl_ijk), 1))), axis=1)
        lbl_xyz[lbl] = nifti.affine.dot(lbl_ijk.T)[:3, :].T

    elec_lbls = []
    for xyz in elec_locs:
        min_dists = np.array([
            np.nanmin(np.sqrt(np.nansum((lbl_xyz[lbl] - xyz)**2, axis=1)))
            for lbl in lbl_xyz
        ])
        elec_lbls.append(lbl_xyz.keys()[np.argmin(min_dists)])
    elec_lbls = np.array(elec_lbls)

    return elec_lbls

    for ii, lbl in enumerate(elec_lbls):
        elec_lbls[ii] = lbl.split('_')[0]
    elec_lbls = np.array(elec_lbls)

    return elec_lbls


def assign_locs_to_lausanne_roi(elec_locs):
    nifti = nib.load('{}/ROIv_scale125_dilated.nii.gz'.format(DIR_LAUSANNE))
    nifti_lbl = pd.read_csv(
        '{}/scale125_with_Yeo2011_7Networks_MNI152.csv'.format(DIR_LAUSANNE))
    nifti_id = np.array(nifti_lbl['Label_ID'], dtype=float)
    nifti_lbl = np.array(['_'.join(i[1][['ROI', 'Hemisphere']]) \
        for i in nifti_lbl.iterrows()])
    nifti_img = nifti.get_data()[:, :, :]

    lbl_xyz = {}
    for lbl, lbl_id in zip(nifti_lbl, nifti_id):
        lbl_ijk = np.array(np.nonzero(nifti_img == lbl_id)).T
        if lbl_ijk.shape[0] == 0:
            print('{}, {} not found.'.format(lbl, lbl_id))
            continue
        lbl_ijk = np.concatenate((lbl_ijk, np.ones((len(lbl_ijk), 1))), axis=1)
        lbl_xyz[lbl] = nifti.affine.dot(lbl_ijk.T)[:3, :].T

    elec_lbls = []
    for xyz in elec_locs:
        min_dists = np.array([
            np.nanmin(np.sqrt(np.nansum((lbl_xyz[lbl] - xyz)**2, axis=1)))
            for lbl in lbl_xyz
        ])
        elec_lbls.append(lbl_xyz.keys()[np.argmin(min_dists)])

    # Strip all excess label terms
    for ii, lbl in enumerate(elec_lbls):
        elec_lbls[ii] = lbl.split('_')[0]
    elec_lbls = np.array(elec_lbls)

    return elec_lbls


def assign_locs_to_yeo_roi(elec_locs):
    nifti = nib.load('{}/ROIv_scale125_dilated.nii.gz'.format(DIR_LAUSANNE))
    nifti_lbl = pd.read_csv(
        '{}/scale125_with_Yeo2011_7Networks_MNI152.csv'.format(DIR_LAUSANNE))
    nifti_id = np.array(nifti_lbl['Label_ID'], dtype=float)
    yeo_lbl = np.array(nifti_lbl['Yeo2011_7Networks'])
    #nifti_lbl = np.array(nifti_lbl['ROI'])
    nifti_lbl = np.array(['_'.join(i[1][['ROI', 'Hemisphere']]) \
        for i in nifti_lbl.iterrows()])
    nifti_img = nifti.get_data()[:, :, :]

    lbl_xyz = {}
    for lbl, lbl_id in zip(nifti_lbl, nifti_id):
        lbl_ijk = np.array(np.nonzero(nifti_img == lbl_id)).T
        if lbl_ijk.shape[0] == 0:
            print('{}, {} not found.'.format(lbl, lbl_id))
            continue
        lbl_ijk = np.concatenate((lbl_ijk, np.ones((len(lbl_ijk), 1))), axis=1)
        lbl_xyz[lbl] = nifti.affine.dot(lbl_ijk.T)[:3, :].T

    elec_lbls = []
    for xyz in elec_locs:
        min_dists = np.array([
            np.nanmin(np.sqrt(np.nansum((lbl_xyz[lbl] - xyz)**2, axis=1)))
            for lbl in lbl_xyz
        ])
        elec_lbls.append(lbl_xyz.keys()[np.argmin(min_dists)])
    elec_lbls = np.array(elec_lbls)

    # For yeo
    for ii, lbl in enumerate(elec_lbls):
        elec_lbls[ii] = yeo_lbl[nifti_lbl == lbl][0]

    return elec_lbls

    for ii, lbl in enumerate(elec_lbls):
        elec_lbls[ii] = lbl.split('_')[0]
    elec_lbls = np.array(elec_lbls)

    return elec_lbls


def assign_locs_to_yeosubg_roi(elec_locs):
    nifti = nib.load('{}/ROIv_scale125_dilated.nii.gz'.format(DIR_LAUSANNE))
    nifti_lbl = pd.read_csv(
        '{}/scale125_with_Yeo2011_7NetworksSubgroup_MNI152.csv'.format(
            DIR_LAUSANNE))
    nifti_id = np.array(nifti_lbl['Label_ID'], dtype=float)
    yeo_lbl = np.array(nifti_lbl['Yeo2011_7Networks'])
    #nifti_lbl = np.array(nifti_lbl['ROI'])
    nifti_lbl = np.array(['_'.join(i[1][['ROI', 'Hemisphere']]) \
        for i in nifti_lbl.iterrows()])
    nifti_img = nifti.get_data()[:, :, :]

    lbl_xyz = {}
    for lbl, lbl_id in zip(nifti_lbl, nifti_id):
        lbl_ijk = np.array(np.nonzero(nifti_img == lbl_id)).T
        if lbl_ijk.shape[0] == 0:
            print('{}, {} not found.'.format(lbl, lbl_id))
            continue
        lbl_ijk = np.concatenate((lbl_ijk, np.ones((len(lbl_ijk), 1))), axis=1)
        lbl_xyz[lbl] = nifti.affine.dot(lbl_ijk.T)[:3, :].T

    elec_lbls = []
    for xyz in elec_locs:
        min_dists = np.array([
            np.nanmin(np.sqrt(np.nansum((lbl_xyz[lbl] - xyz)**2, axis=1)))
            for lbl in lbl_xyz
        ])
        elec_lbls.append(lbl_xyz.keys()[np.argmin(min_dists)])
    elec_lbls = np.array(elec_lbls)

    # For yeo
    for ii, lbl in enumerate(elec_lbls):
        elec_lbls[ii] = yeo_lbl[nifti_lbl == lbl][0]

    return elec_lbls

    for ii, lbl in enumerate(elec_lbls):
        elec_lbls[ii] = lbl.split('_')[0]
    elec_lbls = np.array(elec_lbls)

    return elec_lbls


def contingency_table(p1, p2, plot=False):
    p1 = np.squeeze(p1)
    p2 = np.squeeze(p2)
    assert len(p1) == len(p2)

    id_1 = np.unique(p1)
    id_2 = np.unique(p2)

    # Form contingency table
    T = np.zeros((len(id_1), len(id_2)))
    for i1 in range(len(id_1)):
        for i2 in range(len(id_2)):
            T[i1, i2] = np.sum((p1 == id_1[i1]) & (p2 == id_2[i2]))

    if plot:
        T10 = np.log10(T)
        T10[~np.isfinite(T10)] = 0.0

        plt.figure(figsize=(6, 6), dpi=300.0)
        ax = plt.subplot(111)
        mat = ax.matshow(
            T10,
            aspect=float(2 * T.shape[1]) / float(T.shape[0]),
            cmap='viridis',
            vmin=1.0,
            vmax=2.5)
        cb = plt.colorbar(mat, ax=ax)
        cb.set_label('N Overlapping Electrodes (log)', fontsize=6)
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=5)

        ax.set_xticks(np.arange(T.shape[1]))
        ax.set_xticklabels(id_2, fontsize=5)
        ax.set_xlabel('Module Assignment', fontsize=6)

        ax.set_yticks(np.arange(T.shape[0]))
        ax.set_yticklabels(id_1, fontsize=5)
        ax.set_ylabel('Regional Assignment', fontsize=6)

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.tight_layout()
        plt.savefig(
            './network_modularity_figs/Contingency_Table.Laus_Scale125.pdf',
            transparent=True)
        #plt.savefig('./network_modularity_figs/Contingency_Table.Yeo.pdf', transparent=True)
        plt.show()

    return T


def jaccard_table(model_base_name, gamma='1.1875'):
    df_atlas = pd.read_csv('{}/{}.atlas.csv'.format(DIR_MODULARITY,
                                                    model_base_name))
    df_module = pd.read_csv('{}/{}_full-pos.cons.csv'.format(
        DIR_MODULARITY, model_base_name))

    df_merge = df_atlas.merge(df_module, on='electrode')

    yeo_lbl = df_merge['Yeo_Label'].unique()
    mod_lbl = df_merge['cluster_id'].unique()

    J = np.zeros((len(yeo_lbl), len(mod_lbl)))
    for yi, yy in enumerate(yeo_lbl):
        yy_bool = (df_merge['Yeo_Label'] == yy)
        yy_len = yy_bool.sum()
        for mi, mm in enumerate(mod_lbl):
            mm_bool = (df_merge['cluster_id'] == mm)
            mm_len = mm_bool.sum()

            yy_mm_len = (yy_bool & mm_bool).sum()
            J[yi, mi] = yy_mm_len / (yy_len + mm_len - yy_mm_len)

    return df_merge, J


def run_rand_index_against_lausanne(model_name):
    df_cons = np.load('{}/{}.cons'.format(DIR_MODULARITY, model_name))

    model_obj = se.load('{}/{}.mo'.format(DIR_MODEL, model_name))
    model_locs = np.array(model_obj.get_locs())
    #model_lbls = assign_locs_to_lausanne_roi(model_locs)
    model_lbls = assign_locs_to_yeo_roi(model_locs)

    model_num_lbls = np.zeros(len(model_lbls))
    for ii, ll in enumerate(np.unique(model_lbls)):
        model_num_lbls[model_lbls == ll] = int(ii)
    model_num_lbls = np.asarray(model_num_lbls, int)

    rand_table = {
        'gamma': [],
        'zs_rand': [],
        'rand_null_min': [],
        'rand_null_max': [],
        'rand_null_mean': []
    }

    for gamma in df_cons:
        print(gamma)
        gamma_float = float(gamma)

        cons_lbls = df_cons[gamma][()]

        zs_rand = ncat.zrand(cons_lbls, model_num_lbls)[0]

        rand_null = []
        for ii in range(10):
            rand_null.append(
                ncat.zrand(np.random.permutation(cons_lbls),
                           model_num_lbls)[0])
        rand_null = np.array(rand_null)

        rand_table['gamma'].append(gamma_float)
        rand_table['zs_rand'].append(zs_rand)
        rand_table['rand_null_min'].append(rand_null.min())
        rand_table['rand_null_max'].append(rand_null.max())
        rand_table['rand_null_mean'].append(rand_null.mean())

    # Plot the sweep
    rand_table = pd.DataFrame.from_dict(rand_table)
    rand_table = rand_table.sort_values(by='gamma').reset_index()
    opt_ix = rand_table['zs_rand'].idxmax()
    opt_gamma = rand_table['gamma'].iloc[opt_ix]
    opt_val_max = rand_table['zs_rand'].iloc[opt_ix]
    opt_val_min = rand_table['zs_rand'].min()

    plt.figure(figsize=(6, 6), dpi=300)
    ax = plt.subplot(111)
    ax = sns.lineplot(x='gamma', y='zs_rand', ci=68, data=rand_table, ax=ax)
    ax.vlines(opt_gamma, opt_val_min, opt_val_max, color='r', linewidth=0.5)
    ax.fill_between(
        x=rand_table['gamma'],
        y1=rand_table['rand_null_min'],
        y2=rand_table['rand_null_max'],
        color='k',
        alpha=0.1)
    ax.plot(x=rand_table['gamma'], y=rand_table['rand_null_mean'], color='k')
    ax.set_xlabel('Module Resolution (gamma)')
    ax.set_ylabel('Rand Partition Similarity')
    plt.tight_layout()
    #plt.savefig('./network_modularity_figs/Rand_Sim_Lausanne.{}.pdf'.format(model_name),
    #        transparent=True)
    plt.savefig(
        './network_modularity_figs/Rand_Sim_Yeo.{}.pdf'.format(model_name),
        transparent=True)
    plt.show()

    return rand_table
    """
    # Get a contingency table
    cons_lbls = df_cons[df_cons.keys()[opt_ix]]
    cont_tbl = contingency_table(model_lbls, cons_lbls, plot=True)

    return rand_table, cont_tbl, model_lbls, cons_lbls
    """


def load_all_models(model_base_name):
    full_model = se.Model('{}/{}_full-pos.mo'.format(DIR_MODEL,
                                                     model_base_name))
    full_corr_mat = full_model.get_model()
    full_corr_mat[np.diag_indices_from(full_corr_mat)] = 0
    full_corr_mat = np.arctanh(np.abs(full_corr_mat))
    full_locs = full_model.get_locs()

    dep_model = se.Model('{}/{}_dep-no_mild.mo'.format(DIR_MODEL,
                                                       model_base_name))
    dep_corr_mat = dep_model.get_model()
    dep_corr_mat[np.diag_indices_from(dep_corr_mat)] = 0
    dep_corr_mat = np.arctanh(np.abs(dep_corr_mat))
    dep_corr_mat /= np.sum(dep_corr_mat)

    nodep_model = se.Model('{}/{}_no_dep-no_mild.mo'.format(
        DIR_MODEL, model_base_name))
    nodep_corr_mat = nodep_model.get_model()
    nodep_corr_mat[np.diag_indices_from(nodep_corr_mat)] = 0
    nodep_corr_mat = np.arctanh(np.abs(nodep_corr_mat))
    nodep_corr_mat /= np.sum(nodep_corr_mat)

    df_cons = np.load('{}/{}_full-pos.cons'.format(DIR_MODULARITY,
                                                   model_base_name))

    return full_corr_mat, dep_corr_mat, nodep_corr_mat, df_cons, full_locs


def inter_intra_module_subj(model_base_name, gamma='1.1875'):
    df_cons = np.load('{}/{}_full-pos.cons'.format(DIR_MODULARITY,
                                                   model_base_name))
    massign = df_cons[gamma]
    massign_lbl = np.unique(massign)
    n_massign = len(massign_lbl)
    m_triu_ix, m_triu_iy = np.triu_indices(n_massign, k=0)

    module_feat_dict = {'PID': []}
    for trix, triy in zip(m_triu_ix, m_triu_iy):
        module_feat_dict['{}-{}'.format(trix, triy)] = []

    all_model_name = glob.glob('{}/{}_*.mo'.format(DIR_MODEL, model_base_name))
    pid_list = [
        ll.split('/')[-1].split('.')[0].split('_')[-1] for ll in all_model_name
    ]
    pid_list = [p for p in pid_list if p.isdigit()]
    for pid in pid_list:
        try:
            model = se.Model('{}/{}_{}.mo'.format(DIR_MODEL, model_base_name,
                                                  pid))
        except:
            continue
        print(pid)
        corr_mat = model.get_model()
        corr_mat[np.diag_indices_from(corr_mat)] = 0
        corr_mat = np.abs(corr_mat)

        module_feat_dict['PID'].append(pid)
        for trix, triy in zip(m_triu_ix, m_triu_iy):
            zs = corr_mat[massign == trix, :][:, massign == triy].sum()
            if trix == triy:
                zs /= ((massign == trix).sum() *
                       ((massign == trix).sum() - 1)) * 0.5
            else:
                zs /= ((massign == trix).sum() * ((massign == triy).sum()))

            module_feat_dict['{}-{}'.format(trix, triy)].append(zs)
    module_feat_dict = pd.DataFrame(module_feat_dict)
    module_feat_dict.to_csv(
        '{}/{}_full-pos.sub_module_feat.{:0.2f}.csv'.format(
            DIR_MODULARITY, model_base_name, float(gamma)))


def inter_intra_module_stats(full, depr, nondepr, massign):
    for gamma in ['1.1875', '1.6458333333333333']:
        mm = massign[gamma].copy()

        m_corr = np.zeros((len(np.unique(mm)), len(np.unique(mm))))
        for i1, m1 in enumerate(np.unique(mm)):
            for i2, m2 in enumerate(np.unique(mm)):
                depr_vals = depr[mm == m1, :][:, mm == m2].reshape(-1)
                nondepr_vals = nondepr[mm == m1, :][:, mm == m2].reshape(-1)
                dv = (depr_vals.mean() - nondepr_vals.mean()) / np.sqrt(
                    (depr_vals.var() + nondepr_vals.var()) / 2)
                m_corr[i1, i2] = dv

        m_corr_null = np.zeros((100, len(np.unique(mm)), len(np.unique(mm))))
        for perm in range(100):
            print(perm)
            mm = np.random.permutation(mm)
            for i1, m1 in enumerate(np.unique(mm)):
                for i2, m2 in enumerate(np.unique(mm)):
                    depr_vals = depr[mm == m1, :][:, mm == m2].reshape(-1)
                    nondepr_vals = nondepr[mm == m1, :][:, mm == m2].reshape(
                        -1)
                    dv = (depr_vals.mean() - nondepr_vals.mean()) / np.sqrt(
                        (depr_vals.var() + nondepr_vals.var()) / 2)
                    m_corr_null[perm, i1, i2] = dv

        m_corr_filt = np.nan * m_corr.copy()
        m_corr_filt[((np.abs(m_corr_null) > np.abs(m_corr)).mean(axis=0) <
                     (1 / 100.))] = True

        plt.figure(figsize=(4, 4), dpi=300.0)
        ax = plt.subplot(111)
        mat = ax.matshow(
            m_corr * m_corr_filt, cmap='coolwarm', vmin=-0.6, vmax=0.6)
        cb = plt.colorbar(mat, ax=ax, ticks=[-0.6, 0, 0.6])
        cb.set_label('<---Depressed || Non-Depressed--->', rotation=270)

        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Module Assignment')
        ax.yaxis.set_ticks_position('left')
        ax.set_ylabel('Module Assignment')
        ax.set_title('Intra/Inter Module Correlation')
        plt.tight_layout()
        plt.savefig(
            './network_modularity_figs/inter_intra_module.{:0.2f}.pdf'.format(
                float(gamma)),
            transparent=True)
        plt.show()


def plot_dep_nodep_fc(model_base_name):
    full_corr, dep_corr, nodep_corr, massign, locs = load_all_models(
        model_base_name)
    diff_mat = dep_corr - nodep_corr

    yy = assign_locs_to_yeo_roi(np.array(locs))
    yy_srt_ix = np.argsort(yy)
    yy = yy[yy_srt_ix]
    yy_lbl, yy_lbl_ix = np.unique(yy, return_index=True)

    diff_mat = diff_mat[yy_srt_ix, :][:, yy_srt_ix]

    plt.figure(figsize=(7, 7), dpi=300.0)
    ax = plt.subplot(111)
    mat = ax.matshow(diff_mat, cmap='coolwarm', vmin=-0.2, vmax=0.2)
    ax.vlines(yy_lbl_ix - 0.5, 0, len(yy) - 1, color='k')
    ax.hlines(yy_lbl_ix - 0.5, 0, len(yy) - 1, color='k')

    cb = plt.colorbar(mat, ax=ax)  #, ticks=[-0.6, 0, 0.6])
    cb.set_label('<---Depressed || Non-Depressed--->', rotation=270)

    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(yy_lbl_ix - 0.5)
    ax.set_xticklabels(yy_lbl, rotation=90)
    ax.set_yticks(yy_lbl_ix - 0.5)
    ax.set_yticklabels(yy_lbl)
    ax.yaxis.set_ticks_position('left')
    ax.set_title('Functional Connectivity Across Cognitive Networks')
    plt.tight_layout()
    plt.savefig(
        './network_modularity_figs/dep_nodep_FC_Yeo.pdf', transparent=True)
    plt.show()


def plot_dep_nodep_fc_condense(model_base_name):
    full_corr, dep_corr, nodep_corr, massign, locs = load_all_models(
        model_base_name)
    diff_mat = dep_corr - nodep_corr

    mm = assign_locs_to_yeo_roi(np.array(locs))
    yy_lbl, yy_lbl_ix = np.unique(mm, return_index=True)

    m_corr = np.zeros((len(yy_lbl), len(yy_lbl)))
    for i1, m1 in enumerate(yy_lbl):
        for i2, m2 in enumerate(yy_lbl):
            depr_vals = dep_corr[mm == m1, :][:, mm == m2].reshape(-1)
            nondepr_vals = nodep_corr[mm == m1, :][:, mm == m2].reshape(-1)
            dv = (depr_vals.mean() - nondepr_vals.mean()) / np.sqrt(
                (depr_vals.var() + nondepr_vals.var()) / 2)
            m_corr[i1, i2] = dv

    m_corr_null = np.zeros((100, len(np.unique(mm)), len(np.unique(mm))))
    for perm in range(100):
        print(perm)
        mm = np.random.permutation(mm)
        for i1, m1 in enumerate(np.unique(mm)):
            for i2, m2 in enumerate(np.unique(mm)):
                depr_vals = dep_corr[mm == m1, :][:, mm == m2].reshape(-1)
                nondepr_vals = nodep_corr[mm == m1, :][:, mm == m2].reshape(-1)
                dv = (depr_vals.mean() - nondepr_vals.mean()) / np.sqrt(
                    (depr_vals.var() + nondepr_vals.var()) / 2)
                m_corr_null[perm, i1, i2] = dv

    m_corr_filt = np.nan * m_corr.copy()
    m_corr_filt[((np.abs(m_corr_null) > np.abs(m_corr)).mean(axis=0) <
                 (1 / 100.))] = True

    plt.figure(figsize=(4, 4), dpi=300.0)
    ax = plt.subplot(111)
    mat = ax.matshow(
        m_corr * m_corr_filt, cmap='coolwarm', vmin=-0.4, vmax=0.4)
    cb = plt.colorbar(mat, ax=ax, ticks=[-0.4, 0, 0.4])
    cb.set_label('<---Depressed || Non-Depressed--->', rotation=270)

    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(len(yy_lbl)))
    ax.set_xticklabels(yy_lbl, rotation=90)
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks(np.arange(len(yy_lbl)))
    ax.set_yticklabels(yy_lbl)
    plt.tight_layout()
    plt.savefig(
        './network_modularity_figs/inter_intra_FC_Yeo_condensed.pdf',
        transparent=True)
    plt.show()

    return m_corr, m_corr_filt


def load_all_allegiance(model_base_name):
    full_model = se.Model('{}/{}_full.mo'.format(DIR_MODEL, model_base_name))
    full_locs = full_model.get_locs()

    full_allg = np.load('{}/{}_full.allg'.format(DIR_MODULARITY,
                                                 model_base_name))
    gamma = '0.9583333333333333'
    gamma = '2.333333333333333'
    full_allg_mat = full_allg[gamma][()]['allegiance'] / full_allg[gamma][(
    )]['count']

    depr_allg_mat = np.zeros_like(full_allg_mat)
    nondepr_allg_mat = np.zeros_like(full_allg_mat)
    for sel_df in DF_DEPR.iterrows():
        pt_name = '{}_{}'.format(model_base_name, sel_df[1]['Sub'][2:])
        print(pt_name)

        pt_allg = np.load('{}/{}.allg'.format(DIR_MODULARITY, pt_name))
        pt_allg_mat = pt_allg[gamma][()]['allegiance'] / pt_allg[gamma][(
        )]['count']

        if sel_df[1]['Dep'] == 1:
            depr_allg_mat += pt_allg_mat
        else:
            nondepr_allg_mat += pt_allg_mat
    depr_allg_mat /= (DF_DEPR['Dep'] == 1).sum()
    nondepr_allg_mat /= (DF_DEPR['Dep'] == 0).sum()

    depr_allg_mat[np.diag_indices_from(depr_allg_mat)] = 0
    nondepr_allg_mat[np.diag_indices_from(nondepr_allg_mat)] = 0

    df_cons = np.load('{}/{}_full.cons'.format(DIR_MODULARITY,
                                               model_base_name))
    df_cons = df_cons[gamma]

    return full_allg_mat, depr_allg_mat, nondepr_allg_mat, df_cons, full_locs


def plot_ordered_allegiance(depr, nondepr, locs):
    model_lbls = assign_locs_to_yeo_roi(np.array(locs))
    sort_ix = np.argsort(model_lbls)
    lbl_name, lbl_ix = np.unique(model_lbls[sort_ix], return_index=True)

    diff_allg = (depr - nondepr)[sort_ix, :][:, sort_ix]

    plt.figure(figsize=(6, 6), dpi=100.0)
    ax = plt.subplot(111)
    mat = ax.matshow(diff_allg, cmap='coolwarm', vmin=-0.2, vmax=0.2)
    cb = plt.colorbar(mat, ax=ax, ticks=[-0.2, 0.2])
    cb.set_label('<---Depressed || Non-Depressed--->', rotation=270)

    ax.vlines(
        lbl_ix - 0.5, -0.5, len(model_lbls) - 0.5, color='k', linewidth=1.0)
    ax.hlines(
        lbl_ix - 0.5, -0.5, len(model_lbls) - 0.5, color='k', linewidth=1.0)

    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(lbl_ix)
    ax.set_xticklabels(lbl_name, rotation=90)
    ax.set_xlabel('Electrodes (nodes)')

    ax.yaxis.set_ticks_position('left')
    ax.set_yticks(lbl_ix)
    ax.set_yticklabels(lbl_name)
    ax.set_ylabel('Electrodes (nodes)')

    ax.set_title('Functional Connectivity Across Cognitive Networks')

    plt.show()


def plot_Q_dep(full_model_base_name, dep_model_base_name,
               nodep_model_base_name):
    Q_full = reduce_Q(full_model_base_name)
    Q_dep = reduce_Q(dep_model_base_name)
    Q_nodep = reduce_Q(nodep_model_base_name)
    Q_full = Q_full[(Q_full['gamma'] > 0.9) & (Q_full['gamma'] < 3.1)]
    Q_dep = Q_dep[(Q_dep['gamma'] > 0.9) & (Q_dep['gamma'] < 3.1)]
    Q_nodep = Q_nodep[(Q_nodep['gamma'] > 0.9) & (Q_nodep['gamma'] < 3.1)]

    plt.figure(figsize=(4, 4), dpi=300)
    ax = plt.subplot(111)
    ax = sns.lineplot(
        x='gamma', y='Q', data=Q_dep, ci=68, ax=ax, linewidth=0.5)
    ax = sns.lineplot(
        x='gamma', y='Q', data=Q_nodep, ci=68, ax=ax, linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Modularity (Q)')
    ax.legend(['Depressed Model', 'Non-Depressed Model'])
    plt.tight_layout()
    plt.savefig('./network_modularity_figs/Q_dep_nodep.pdf', transparent=True)
    plt.show()

    return Q_dep, Q_nodep


def write_gefx(model_base_name, gamma='1.1875'):
    full_corr, dep_corr, nodep_corr, massign, locs = load_all_models(
        model_base_name)

    G_full = nx.from_numpy_matrix(full_corr)
    G_dep = nx.from_numpy_matrix(dep_corr)
    G_nodep = nx.from_numpy_matrix(nodep_corr)

    mmm = {}
    for m_i, m in enumerate(massign[gamma]):
        mmm[m_i] = int(m)
    nx.set_node_attributes(G_full, mmm, 'module')
    nx.set_node_attributes(G_dep, mmm, 'module')
    nx.set_node_attributes(G_nodep, mmm, 'module')

    try:
        nx.write_gexf(G_full, '{}/{}_full-pos.gexf'.format(
           DIR_MODULARITY, model_base_name))

        nx.write_gexf(G_dep, '{}/{}_dep-no_mild.gexf'.format(
           DIR_MODULARITY, model_base_name))

        nx.write_gexf(G_nodep, '{}/{}_no_dep-no_mild.gexf'.format(
           DIR_MODULARITY, model_base_name))
    except:
        pass

    return G_full
