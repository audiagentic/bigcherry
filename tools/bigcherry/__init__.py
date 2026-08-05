"""bigcherry — release-tolerant HIP autotune overlay for llama.cpp."""

__version__ = "0.1.0"

# Bumped whenever the on-disk shape of an artifact changes in a way that makes
# older artifacts unreadable. Distinct from the llama.cpp source revision and
# from the candidate manifest hash; all three form the build namespace.
ARTIFACT_VERSION = 1
