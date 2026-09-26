"""BigCherry durable job-service execution adapters.

Domain/scientific policy must remain outside backend-specific adapters. In
particular, only ``bigcherry.jobs.slurm`` may import/encode Slurm CLI concepts.
"""
