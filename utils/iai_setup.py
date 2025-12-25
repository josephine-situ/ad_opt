"""
IAI (Interpretable AI) initialization and configuration.

This module handles IAI library setup, including:
- Detecting cluster vs local environment
- Setting appropriate system image paths
- Configuring Julia threads
- Importing interpretableai (iai)

Usage:
    from utils.iai_setup import iai
    # Now use iai normally
"""

import os
import sys


def setup_iai():
    """
    Initialize IAI with appropriate configuration for current environment.
    
    Returns:
    - iai: The interpretableai module
    
    Raises:
    - ImportError: If interpretableai cannot be imported after setup
    """
    
    # Check if running on Slurm cluster (Engaging cluster)
    if 'SLURM_NODEID' in os.environ:
        # Engaging cluster configuration
        threads = os.getenv('SLURM_CPUS_PER_TASK')
        if threads:
            os.environ['JULIA_NUM_THREADS'] = threads
            print(f"[IAI] Cluster mode: Using {threads} Julia threads")
        
        os.environ['IAI_JULIA'] = "/orcd/software/community/001/pkg/julia/1.10.4/bin/julia"
        os.environ['IAI_SYSTEM_IMAGE'] = os.path.expanduser("~/iai/sys.so")
        os.environ['IAI_DISABLE_COMPILED_MODULES'] = "true"
        print("[IAI] Configured for Engaging cluster")
    else:
        # Local machine configuration (Windows)
        os.environ['IAI_SYSTEM_IMAGE'] = "C:\\Users\\jsitu\\IAI\\sys.dll"
        print("[IAI] Configured for local Windows machine")
    
    # Import interpretableai
    try:
        from interpretableai import iai
        print("[IAI] Successfully imported interpretableai")
        return iai
    except ImportError as e:
        print(f"[IAI] ERROR: Could not import interpretableai")
        print(f"[IAI] Install with: pip install iai")
        print(f"[IAI] Note: IAI requires a valid license")
        print(f"[IAI] Details: {e}")
        raise


# Initialize on module import
try:
    iai = setup_iai()
except ImportError:
    sys.exit(1)
