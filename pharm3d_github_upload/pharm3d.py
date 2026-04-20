from webutils import search_pend_jobs
import pandas as pd
import json
result=search_pend_jobs()
jobid= result[2]
parameters = json.loads(result[5])
print(jobid, parameters)