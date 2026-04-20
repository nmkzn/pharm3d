#!/home/shiyu/anaconda3/bin/python
from embed import restore_mlp, write_log, plot_fig
from embed import get_protein_ligand_neighbors, featurizer_new, read_smiles, vertices_gen, pharm_train
from sqlutils import search_pend_jobs, alter_pend_jobs
import os, torch, json, time

ncpu=10
torch.set_num_threads(ncpu)
os.environ ['OMP_NUM_THREADS'] = str(ncpu)
os.environ["MKL_NUM_THREADS"] = str(ncpu)
while True:
    job = search_pend_jobs(col='pending', jobtype='model')
    if job == None:
        time.sleep(20 * 60) #sleep 20min
        continue
    jobid = job[2]
    dirname=f"../static/jobfolder/{jobid}/"
    alter_pend_jobs(jobid=jobid, status="computing")
    try: 
        parameters = json.loads(job[5])
        num_confs = int(parameters['conformer'])
        write_log(dirname, f"Starting job '{dirname}' ......")
        pocket_path_npy = get_protein_ligand_neighbors(dirname,ligand_residue_id='UNK',cutoff_distance=5)
        write_log(dirname, f"Pocket process done  ......")
        min_x, max_x, min_y, max_y, min_z, max_z, df = read_smiles(dirname, num_confs=num_confs, ncpu=ncpu)
        write_log(dirname, f"Read molecules done ......")
        vertices = vertices_gen(pocket_path_npy, min_x, max_x, min_y, max_y, min_z, max_z, ncpu=ncpu)
        ind_att_pd_path, tensor_path, state_path = featurizer_new(dirname, min_x, max_x, min_y, max_y, min_z, max_z, df, vertices, ncpu=ncpu)
        write_log(dirname, f"Features generation done ......")

        write_log(dirname, "Starting model training ......")
        accuracy_path = pharm_train(dirname=dirname, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, min_z=min_z, max_z=max_z, tensor_path=tensor_path, state_path=state_path)
        write_log(dirname, "Starting generate pharmacophores ......")
        for fold in range(5):
            model_path = f"mymodel_{fold}fold.pth"
            pdb_path = restore_mlp(dirname=dirname, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, min_z=min_z, max_z=max_z, ind_att_pd_path=ind_att_pd_path, model_path=model_path, nfeat=20)
            pdb_path = restore_mlp(dirname=dirname, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, min_z=min_z, max_z=max_z, ind_att_pd_path=ind_att_pd_path, model_path=model_path, nfeat=100)
            plot_fig(dirname=dirname, loss_path=f'Fold{fold}_loss.txt', title=f'Fold_{fold}', loss_fig=f'Fold{fold}_loss.png')
        alter_pend_jobs(jobid=jobid, status="success")
    except Exception as e:
        write_log(dirname, str(e))
        alter_pend_jobs(jobid=jobid, status="failed")
