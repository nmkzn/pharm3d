import json, time
from screen import screen_read_mols, screen_match, write_log
from sqlutils import search_pend_jobs, alter_pend_jobs

ncpu=5
while True:
    job = search_pend_jobs(col='pending', jobtype='screen')
    if job == None:
        time.sleep(20 * 60) #sleep 20min
        continue
    jobid = job[2]
    dirname=f"../static/jobfolder/{jobid}/"
    alter_pend_jobs(jobid=jobid, status="computing")
    try:
        parameters = json.loads(job[5])
        num_confs = int(parameters['conformer'])
        nfeat = int(parameters['nfeat'])
        write_log(dirname, f"Starting screen job '{dirname}' ......")
        screen_read_mols(dirname, num_confs=num_confs, ncpu=ncpu)
        screen_match(dirname, weight='indicesAttWtPd.csv', ncpu=ncpu, nfeat=nfeat)
        alter_pend_jobs(jobid=jobid, status="success")
    except Exception as e:
        write_log(dirname, str(e))
        alter_pend_jobs(jobid=jobid, status="failed")