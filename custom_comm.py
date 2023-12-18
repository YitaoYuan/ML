import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import socket
import pickle
import psutil
import queue# this is for multi-thread, not multi-process
import threading
import time
import numpy as np

# from multiprocessing import Process, Queue
device = None

def prefix_len(x, y):
    ans = 0
    for i, j in zip(x, y):
        if i != j:
            break
        ans += 1
    return ans

def lpm_ip_address(master_ip): # longest prefix matching
    # Get all network interfaces
    addr = [a.address for ifname, ifaddrs in psutil.net_if_addrs().items() for a in ifaddrs if a.family == socket.AF_INET]
    pref = [prefix_len(a, master_ip) for a in addr]
    max_pref = max(pref)
    return addr[pref.index(max_pref)]

def tcp_create_group():
    assert dist.is_initialized()
    interfaces = psutil.net_if_addrs()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size > 1
    global device
    device = torch.device("cuda:{}".format(local_rank))

    server_address = (os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]) + 1)
    print("Rank 0 address:", server_address)

    recv_listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    recv_ip = lpm_ip_address(server_address[0])
    print("Local recv IP:", recv_ip)
    recv_addr = (recv_ip, 0)
    recv_listen_socket.bind(recv_addr) 
    recv_listen_socket.listen(1)
    recv_addr = recv_listen_socket.getsockname()
    
    if rank == 0:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(server_address)
        server_socket.listen(world_size)

        client = [None] * world_size
        for i in range(1, world_size):
            client_socket, client_address = server_socket.accept()
            data = client_socket.recv(1500)
            client_rank, client_recv_addr = pickle.loads(data)
            client[client_rank] = (client_socket, client_recv_addr)
        client[0] = (None, recv_addr)

        for i in range(1, world_size):
            curr_socket = client[i][0]
            next_addr = client[(i+1)%world_size][1]
            curr_socket.send(pickle.dumps(next_addr))

        rem_recv_addr = client[1][1]

        print("Connection listen addresses:")
        for _, a in client:
            print(a)
    else:
        # tell rank 0 my recv_addr
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5)
        client_socket.connect(server_address)
        data = (rank, recv_addr)
        client_socket.send(pickle.dumps(data))
        # get the recv_addr of the next member
        data = client_socket.recv(1500)
        rem_recv_addr = pickle.loads(data)

    # connect to the next member
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for i in range(2):
        if rank % 2 == i:
            recv_socket, rem_send_addr = recv_listen_socket.accept()
        else:
            send_socket.connect(rem_recv_addr)
    send_addr = send_socket.getsockname()

    print(f"Rank {rank} connections: {rem_send_addr} -> {recv_addr} == {send_addr} -> {rem_recv_addr}")

    # start background process
    process_group = (send_socket, recv_socket, queue.Queue(), threading.Condition())
    start_allreduce_process(process_group)

    return process_group

def tcp_destroy_group(process_group):
    stop_allreduce_process()

# preindex = 0
def tcp_allreduce(process_group, bucket): # the name cannot be changed
    send_socket, recv_socket, work_queue, cond = process_group
    fut = torch.futures.Future()
    # fut.set_result(bucket.buffer())
    # idx = bucket.index()
    # global preindex
    # if idx != 0 and idx != preindex + 1:
    #     raise "index not match"
    # preindex = idx
    work_queue.put((bucket, fut)) # this is thread-safe even without cond
    with cond:
        cond.notify()
    return fut

daemon = None
daemon_stop = False
def start_allreduce_process(process_group):
    global daemon
    daemon = threading.Thread(target=allreduce_process, args=(process_group, ))
    daemon.start()

def stop_allreduce_process():
    global daemon_stop
    daemon_stop = True

def allreduce_process(process_group):
    global daemon_stop
    send_socket, recv_socket, work_queue, cond = process_group
    while not daemon_stop:
        with cond:
            cond.wait(timeout = 1)
        if work_queue.empty():
            continue
        bucket, fut = work_queue.get() # this is thread-safe even without cond
        send_buf = bucket.buffer()
        do_allreduce(send_buf, process_group)
        fut.set_result(send_buf)
    print("TCP Allreduce Daemon Exit")

def tensor_to_bytes(data):
    return data.cpu().numpy().tobytes()

def bytes_to_tensor_32(data):
    return torch.tensor(np.frombuffer(data, dtype=np.float32)).to(device)

def do_allreduce(tensor, process_group): # a naive chain allreduce
    send_socket, recv_socket, work_queue, cond = process_group
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # data = pickle.dumps(tensor) # CPU?
    data = tensor_to_bytes(tensor)
    size = len(data)
    if rank != 0:
        # tmp_tensor = pickle.loads(recv_socket.recv(size, socket.MSG_WAITALL)) # assert this is on the same device
        data = recv_socket.recv(size, socket.MSG_WAITALL)
        tmp_tensor = bytes_to_tensor_32(data)
        tensor.add_(tmp_tensor) # put
        # data = pickle.dumps(tensor)
        data = tensor_to_bytes(tensor)
    
    if rank != world_size-1:
        send_socket.sendall(data)
        data = send_socket.recv(size, socket.MSG_WAITALL) 
        # tmp_tensor = pickle.loads(recv_socket.recv(size, socket.MSG_WAITALL)) # assert this is on the same device
        tmp_tensor = bytes_to_tensor_32(data)
        tensor.copy_(tmp_tensor) # put

    if rank != 0:
        recv_socket.sendall(data)

    

    

