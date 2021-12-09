## Errors

#### Network Interfacing
**ERROR:** RuntimeError: [enforce fail at ../third_party/gloo/gloo/transport/tcp/device.cc:83] ifa != nullptr. Unable to find address for: vx00  
**REASON:** You mention 2 machines, so using localhost isn't going to cut it. I expect the error to come from machine A trying to connect to machine B over localhost, which obviously won't work. By default, PyTorch will use the machine's hostname to figure out its external IP address. It is likely that the hostname of the machines you're using both resolve to ::1 instead. To override this behavior, you can set the GLOO_SOCKET_IFNAME environment variable to the name of the network interface that connects these machines (for example GLOO_SOCKET_IFNAME=eth0)  
**FIX:** First run ifconfig to find the names of the kernel-residnet ACTIVE network interfaces. WM HPC apparently doesn't have an eth0 this list displayed with ifconfig is the equivalent. Choose 1 (you stuck with 'eno1' for no specific reason). Then in code set ***os.environ['GLOO_SOCKET_IFNAME'] = "eno1"***  
**ORIGINAL ERROR LOG:** user=hmbaier/sciclone/home20/hmbaier/test_rpc/errors/network_interfacing_error.e9364662  
**LINKS:**  
    1. https://github.com/facebookresearch/PyTorch-BigGraph/issues/119  
    2. https://pytorch.org/docs/stable/distributed.html  
    3. https://linux.die.net/man/8/ifconfig  

#### Local Value
**ERROR:** AttributeError: 'list' object has no attribute 'local_value'  
**REASON:** Missing 'to' argument in the rpc_async call  
**FIX:** ob_rref needs to be second arguemnt in rpc_asymc call after refeerence to funcction and before function specfic arguments  
**ORIGINAL ERROR LOG:** user=hmbaier/sciclone/home20/hmbaier/test_rpc/errors/local_value_error.e9364662  


#### torchrun
**ERROR:** /sciclone/home20/hmbaier/.conda/envs/dhsrl4/bin/python: No module named torchrun  
**REASON:** Incorrect version of pytorch  
**FIX**:  
```
[meltemi] conda create -n dhsrl4  
[meltemi] conda activate dhsrl4  
[meltemi] conda install pytorch torchvision torchaudio cpuonly -c pytorch  
```
*within job file, run training script without calling python first:*  
```
torchrun /sciclone/home20/hmbaier/test_rpc/test.py
```
**ORIGINAL ERROR LOG:** user=hmbaier/sciclone/home20/hmbaier/test_rpc/errors/torchrun_error.e9364662  



#### Running multi-node job hangs
**ERROR:** No specfic eror, job simply hangs when attempting to start  
**REASON:** I think this is because you were using the c1od Rendezvous backend. You swithced to the etcd-v2 backend and ran the following installs:  

```
conda install -c conda-forge etcd
conda install -c anaconda urllib3
pip install python-etcd

torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=790875 --rdzv_backend=etcd-v2 rdzv_endpoint=mlt001.sciclone.wm.edu:29400 /sciclone/home20/hmbaier/test_rpc/test.py
```   
**ORIGINAL ERROR LOG:** Not really relevant here


#### In operator() at tensorpipe
**ERROR:** RuntimeError: In operator() at tensorpipe/common/ibv.h:172 "": Invalid argument  
**REASON:** Your interface's name, hfi, seems to be an "Intel® Omni-Path Host Fabric Interface Adapters". This is literally the first time I hear about such a device. Though it seems it suffers from the same issues that affect the EFA devices on AWS: it claims it can be used as an InfiniBand device, and this "tricks" TensorPipe into trying to using it as such, but it doesn't support some of the features that TensorPipe requires.  
TensorPipe is a tensor-aware channel to transfer rich objects from one process to another while using the fastest transport for the tensors contained therein (e.g., CUDA device-to-device copy).  
Part of this was needing (I think) to switch from an ethernet switch to an InfiniBand switch  
**FIX:** 
Step 1: Remove
```
# os.environ['GLOO_SOCKET_IFNAME'] = "ib0"  
```
Step 2: Change TP SOCKET_IFNAME to  
```
os.environ['TP_SOCKET_IFNAME'] = "ib0"
```
Step 3: Change init_rpc line to (i.e. fix the backend options):
```
rpc.init_rpc(AGENT_NAME, rank = rank, world_size = world_size, rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout=20))
```
**LINKS:**
    1. https://issueexplorer.com/issue/pytorch/tensorpipe/413  