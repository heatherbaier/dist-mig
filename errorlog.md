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
