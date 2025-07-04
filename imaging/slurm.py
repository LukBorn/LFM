import pathlib, tempfile, re, subprocess, functools
import dill, time


class PythonExecutorSLURM:
    """ Execute python code on a SLURM-managed cluster

    Args:
        job_path (str): where to store job files
        cmd_prefix (str): prefix for sbatch command. Use this to set up ssh login, if needed.
        **kwargs: SLURM parameters. Use underscores (_) instead of hyphens (-)
    """

    def __init__(self, job_path, conda_env=None, cmd_prefix=[], exclusive=True, **kwargs):

        self.job_path = job_path
        self.sbatch_cmd = cmd_prefix + ['sbatch']
        self.conda_env = conda_env
        self.slurm_params = dict()
        self.exclusive = exclusive
        for k in kwargs.keys():
            self.slurm_params[k.replace('_', '-')] = kwargs[k]

    def sbatch_string_python(self, python_code):
        """ Create batch script string, given python code and conda env

        Args:
            python_code (str):
            conda_env (str): 
        """
        slurm_params_str = ''.join([f'#SBATCH --{k}={v}\n' for k, v in self.slurm_params.items()])
        if self.exclusive:
            slurm_params_str += '#SBATCH --exclusive\n'
        if self.conda_env is None:
            conda_str = ''
        else:
            #conda_str = f'eval "$(conda shell.bash hook)"\nconda activate {self.conda_env}'
            conda_str = f'source ~/.bashrc \nconda activate {self.conda_env}'
            
        sbatch_string = '\n'.join([
            f'#!/bin/bash',
            f'#SBATCH --error={self.job_path}/%j.err',
            f'#SBATCH --output={self.job_path}/%j.out',
            f'#SBATCH --open-mode=append',
            slurm_params_str,
            f'scontrol write batch_script $SLURM_JOB_ID {self.job_path}/$SLURM_JOB_ID.sh',
            conda_str,
            f'srun --output {self.job_path}/%j.out --error {self.job_path}/%j.err --unbuffered python <<"EOF"',
            python_code,
            f'EOF',
            #f'echo "< reached end of batch script >"',
        ])
        return sbatch_string

    def submit_code(self, python_code):
        """ Submit python code (multiline string) for execution on the cluster

        Args:
            python_code (str):
            conda_env (str): 

        Returns: 
            (int): SLURM job id
        """
        sbatch_string = self.sbatch_string_python(python_code)
        pathlib.Path(self.job_path).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', dir=self.job_path) as tfile:
            tfile.write(sbatch_string)
            tfile.flush()
            process = subprocess.Popen(self.sbatch_cmd + [tfile.name], stdout=subprocess.PIPE)
            output, error = process.communicate()
        print(output.decode('utf-8'))
        job_id = int(re.findall('\d+', str(output))[0])
        return job_id

    def submit(self, func, *args, **kwargs):
        """ Submit python callabe for execution on the cluster. 

        Args:
            fun (callable): callable being submitted (uses dill for pickling).
             *args, **kwargs: arguments passed to function. Remember that everything will be pickled, keep it small.

        Returns: 
            (int): SLURM job id
        """
        if len(args)+len(kwargs) > 0:
            func = functools.partial(func, *args, **kwargs)
        pickled = dill.dumps(func)
        #pickled = base64.b64encode(lzma.compress(dill.dumps(execute_this))) #unpickled = dill.loads(lzma.decompress(base64.b64decode(pickle_encoded)))
        if len(pickled) > 1e4:
            raise Exception('pickle too large!')
        python_code = '\n'.join([
            '',
            f'pickled = {pickled}',
            f'',
            f'import dill',
            f'unpickled = dill.loads(pickled)',
            f'unpickled()',
        ])
        job_id = self.submit_code(python_code)
        return job_id
    

class SlowProgressLogger:
    def __init__(self, total, description="Progress", update_interval=5):
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
    def update(self, increment=1):
        self.current += increment
        current_time = time.time()
        
        # Only update log periodically
        if (current_time - self.last_update_time >= self.update_interval or 
            self.current == self.total):
            
            elapsed = current_time - self.start_time
            items_per_sec = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / items_per_sec if items_per_sec > 0 else 0
            
            print(f"{self.description}: {self.current}/{self.total} ({self.current/self.total*100:.1f}%) "
                  f"- {items_per_sec:.2f} it/s - {time.strftime("%H:%M:%S", time.gmtime(eta))} remaining")
                  
            self.last_update_time = current_time
            