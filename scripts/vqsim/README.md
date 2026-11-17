# vqsim and the VQ study

- `vqsim.c`: the C simulator of the virtual-queue (VQ) design space, exact with `src/vista/benchmarks/cachelab/sandbox/vq.py`. It has three modes: cold, handover (S3-FIFO serves the prefix), and online (the live cache switches design; miniature simulations or a given schedule). Build it with `cc -O2 -o vqsim vqsim.c`; the usage is at the top of the file and in `run_online`.
- `study/`: the host scripts behind the version-1 results, kept as they ran on the node (paths `/mydata/vq`, `/mydata/vqsim3`).
- `v1/`: the frozen version-1 portfolio, the seed designs, the board picks, the ghost-depth sweep and the aggregate logs. Logs carry no trace names.

The write-up is in `docs/cachelab-v1.md`.
