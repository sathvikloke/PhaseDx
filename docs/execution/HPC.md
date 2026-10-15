# Local and optional Einstein execution

Local CPU/MPS/GPU execution is the default where supported. E05 prepares an HPC alternative; it is not a required delay for CPU/source tasks and does not authorize job submission.

The project has historical Einstein HPC4 connection instructions. Reuse only the user's existing authenticated SSH master, confirm actual PhaseDx repository/remote directory/environment, and identify existing project jobs by recorded job IDs plus working directory. Account-wide jobs may belong to other projects.

Read-only session check on the user's configured machine:

```sh
ssh -O check sloke.hpc.einsteinmed.edu
ssh -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false sloke.hpc.einsteinmed.edu "bash -lc 'sinfo --clusters=HPC4'"
ssh -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false sloke.hpc.einsteinmed.edu "bash -lc 'squeue --clusters=HPC4 --user=sloke'"
```

If the shared master is absent, record the missing connection and let the user restore it. Do not change DNS/VPN/SSH security, enter passwords or recreate the shared master. The historical IP is not a permanent configuration. No sensitive socket, credential or remote data paths are needed in public files.

Before any separately authorized submission, record the actual remote project root, environment, staging permissions, input identities, partition/GRES, GPU count, memory, wall time and cumulative project allocation. An available GPU is not a reservation. The resource values are deliberately **not invented in this backlog**. Use bounded Slurm batch jobs, not login-node computation; preserve submission scripts, job IDs, raw logs, failures and accounting. Check existing project jobs before retrying.

E23/E24 inference shards are possible HPC candidates. Metadata/source work and deterministic scoring can remain local. Do not launch new training, refits, architecture searches or other-project workloads just because HPC access works. H02 governs the actual run scope; E04 records real concurrency and resumes only matching completed shards.
