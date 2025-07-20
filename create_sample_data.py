"""
Create sample cluster performance data for testing the ML pipeline.
This script generates synthetic data that mimics the structure of real cluster data.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime

def create_sample_data(n_samples=1000):
    """Create sample cluster performance data."""
    
    np.random.seed(42)
    random.seed(42)
    
    # Cluster configuration options
    cluster_types = ['self-managed', 'managed', 'hosted']
    arch_types = ['amd64', 'arm64']
    platforms = ['AWS', 'Azure', 'GCP', 'BareMetal']
    regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
    sdn_types = ['OVNKubernetes', 'OpenShiftSDN']
    instance_types = ['m5.xlarge', 'm5.2xlarge', 'm6a.xlarge', 'r5.xlarge']
    
    data = []
    
    for i in range(n_samples):
        # Basic cluster configuration
        cluster_type = random.choice(cluster_types)
        platform = random.choice(platforms)
        control_plane_arch = random.choice(arch_types)
        worker_arch = random.choice(arch_types)
        
        # Node counts (correlated with performance)
        master_nodes = random.choice([3, 5])
        worker_nodes = random.randint(3, 50)
        infra_nodes = random.randint(0, 5)
        total_nodes = master_nodes + worker_nodes + infra_nodes
        
        # Job configuration
        qps = random.randint(10, 100)
        burst = random.randint(50, 500)
        job_iterations = random.randint(1, 10)
        
        # Performance metrics (synthetic but realistic)
        base_cpu_load = 0.1 + (worker_nodes / 100) + random.gauss(0, 0.05)
        base_memory_load = 100 + (worker_nodes * 5) + random.gauss(0, 20)
        
        # CPU metrics
        cpu_kubelet = max(0.01, base_cpu_load + random.gauss(0, 0.02))
        max_cpu_kubelet = cpu_kubelet * random.uniform(1.5, 3.0)
        cpu_masters = max(0.01, base_cpu_load * 0.8 + random.gauss(0, 0.01))
        max_cpu_masters = cpu_masters * random.uniform(2.0, 4.0)
        cpu_workers = max(0.01, base_cpu_load * 1.2 + random.gauss(0, 0.02))
        max_cpu_workers = cpu_workers * random.uniform(1.8, 3.5)
        
        # Memory metrics (in MB)
        memory_kubelet = max(50, base_memory_load + random.gauss(0, 10))
        max_memory_kubelet = memory_kubelet * random.uniform(1.5, 2.5)
        memory_masters = max(100, base_memory_load * 2 + random.gauss(0, 50))
        max_memory_masters = memory_masters * random.uniform(1.3, 2.0)
        memory_workers = max(80, base_memory_load * 1.5 + random.gauss(0, 30))
        max_memory_workers = memory_workers * random.uniform(1.4, 2.2)
        
        # API latency metrics (in milliseconds)
        base_latency = 0.005 + (total_nodes / 1000) + (qps / 10000)
        avg_ro_latency = max(0.001, base_latency + random.gauss(0, 0.002))
        max_ro_latency = avg_ro_latency * random.uniform(3.0, 10.0)
        avg_mutating_latency = max(0.001, base_latency * 2 + random.gauss(0, 0.005))
        max_mutating_latency = avg_mutating_latency * random.uniform(2.0, 8.0)
        
        # ETCD metrics
        etcd_disk_commit = max(0.001, 0.003 + random.gauss(0, 0.001))
        etcd_wal_fsync = max(0.001, 0.005 + random.gauss(0, 0.002))
        etcd_round_trip = max(0.001, 0.002 + random.gauss(0, 0.0005))
        
        row = {
            # Metadata (input features)
            'clusterType': cluster_type,
            'controlPlaneArch': control_plane_arch,
            'workerArch': worker_arch,
            'platform': platform,
            'region': random.choice(regions),
            'sdnType': random.choice(sdn_types),
            'masterNodesCount': master_nodes,
            'workerNodesCount': worker_nodes,
            'infraNodesCount': infra_nodes,
            'totalNodes': total_nodes,
            'masterNodesType': random.choice(instance_types),
            'workerNodesType': random.choice(instance_types),
            'infraNodesType': random.choice(instance_types) if infra_nodes > 0 else '',
            'k8sVersion': f"v1.{random.randint(25, 30)}.{random.randint(0, 12)}",
            'ocpVersion': f"4.{random.randint(12, 16)}.{random.randint(0, 20)}",
            'passed': random.choice([True, False]),
            'jobConfig.qps': qps,
            'jobConfig.burst': burst,
            'jobConfig.jobIterations': job_iterations,
            'jobConfig.cleanup': random.choice([True, False]),
            'jobConfig.waitForDeletion': random.choice([True, False]),
            
            # Performance metrics (target outputs)
            'cpu-kubelet': round(cpu_kubelet, 6),
            'max-cpu-kubelet': round(max_cpu_kubelet, 6),
            'cpu-masters': round(cpu_masters, 6),
            'max-cpu-masters': round(max_cpu_masters, 6),
            'cpu-workers': round(cpu_workers, 6),
            'max-cpu-workers': round(max_cpu_workers, 6),
            'memory-kubelet': round(memory_kubelet, 2),
            'max-memory-kubelet': round(max_memory_kubelet, 2),
            'memory-masters': round(memory_masters, 2),
            'max-memory-masters': round(max_memory_masters, 2),
            'memory-workers': round(memory_workers, 2),
            'max-memory-workers': round(max_memory_workers, 2),
            'avg-ro-apicalls-latency': round(avg_ro_latency, 6),
            'max-ro-apicalls-latency': round(max_ro_latency, 6),
            'avg-mutating-apicalls-latency': round(avg_mutating_latency, 6),
            'max-mutating-apicalls-latency': round(max_mutating_latency, 6),
            '99thEtcdDiskBackendCommit': round(etcd_disk_commit, 6),
            '99thEtcdDiskWalFsync': round(etcd_wal_fsync, 6),
            '99thEtcdRoundTripTime': round(etcd_round_trip, 6),
            'max-99thEtcdDiskBackendCommit': round(etcd_disk_commit * random.uniform(1.5, 3.0), 6),
            'max-99thEtcdDiskWalFsync': round(etcd_wal_fsync * random.uniform(1.5, 3.0), 6),
            'max-99thEtcdRoundTripTime': round(etcd_round_trip * random.uniform(2.0, 5.0), 6),
        }
        
        data.append(row)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Creating sample cluster performance data...")
    
    # Create sample data
    df = create_sample_data(1000)
    
    # Save to CSV
    output_path = "data/raw/cluster_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Sample data created and saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nSample data creation completed!")
