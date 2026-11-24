/* vqonline_v1.h: the frozen version-1 candidates (scripts/vqsim/v1/portfolio.json; OnlineV1 in
 * sandbox/vq.py), in order. A test checks this file against portfolio.json. */
#ifndef VQONLINE_V1_H
#define VQONLINE_V1_H
#define VQO_V1_K 32
static const char *const VQO_V1_CANDIDATES[VQO_V1_K] = {
    "maxf=3 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0",
    "maxf=7 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 asmall=0 amain=1 st=0.03,1,2,1,0,-1,1,1.0,0.0 st=0,1,1,1,2,-1,0,0.0,0.0",
    "maxf=3 entry=0 overflow=2 ghost=0.9 gentry=1 gfreq=1 asmall=0 amain=2 st=0.15,0,0,1,0,-1,0,0.0,0.0 st=0.15,1,1,2,0,-1,1,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0",
    "maxf=5 entry=0 overflow=4 ghost=0.5 gentry=1 gfreq=0 st=0.2,1,2,4,0,1,0 st=0.01,0,0,2,1,-1,0 st=0.2,0,0,4,1,-1,0 st=0.6,1,1,3,0,4,0 st=0,1,1,4,0,-1,0",
    "maxf=1 entry=0 overflow=2 ghost=2.0 gentry=1 gfreq=0 asmall=0 amain=2 st=0.05,1,2,2,0,-1,1,0.0,0.0 st=0.3,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0",
    "maxf=7 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 asmall=0 amain=1 st=0.01,1,2,1,0,-1,1,1.0,0.0 st=0,1,2,1,2,-1,0,0.0,0.0",
    "maxf=1 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 asmall=0 amain=1 st=0.3,1,2,1,0,-1,1,0.0,0.0 st=0,1,1,1,2,-1,0,1.0,0.0",
    "maxf=3 entry=0 overflow=2 ghost=0.9 gentry=2 gfreq=0 st=0.025,0,0,1,1,-1,0 st=0.025,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0",
    "maxf=3 entry=0 overflow=2 ghost=0.5 gentry=2 gfreq=0 st=0.1,0,0,1,1,-1,0 st=0.1,1,2,2,0,-1,1 st=0,1,1,2,2,-1,0",
    "maxf=2 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 st=0.3,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0",
    "maxf=3 entry=0 overflow=2 ghost=0.9 gentry=1 gfreq=0 asmall=0 amain=2 st=0.05,1,2,1,0,-1,1,0.0,0.0 st=0.1,1,1,2,0,-1,1,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0",
    "maxf=3 entry=0 overflow=1 ghost=0.5 gentry=0 gfreq=1 asmall=0 amain=1 st=0.05,1,2,1,0,-1,1,0.0,0.0 st=0,1,1,1,2,-1,0,0.0,0.0",
    "maxf=3 entry=0 overflow=3 ghost=0.9 gentry=0 gfreq=1 st=0.02,1,2,1,0,-1,1 st=0.02,1,2,3,0,2,0 st=0.05,1,1,3,0,-1,1 st=0,1,1,3,2,-1,0",
    "maxf=3 entry=0 overflow=2 ghost=0.9 gentry=2 gfreq=0 st=0.1,1,2,1,0,-1,1 st=0.01,1,3,2,0,-1,1 st=0,1,1,2,2,-1,0",
    "maxf=3 entry=0 overflow=2 ghost=0.5 gentry=2 gfreq=0 st=0.025,0,0,1,1,-1,0 st=0.025,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0",
    "maxf=1 entry=0 overflow=2 ghost=1.8 gentry=1 gfreq=0 asmall=0 amain=2 st=0.08,1,2,1,0,-1,1,0.0,0.0 st=0.3,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0",
    "maxf=2 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 st=0.05,1,3,1,0,-1,1 st=0,1,1,1,2,-1,0",
    "maxf=3 entry=0 overflow=2 ghost=1.5 gentry=1 gfreq=0 asmall=0 amain=2 st=0.1,1,2,1,0,-1,1,0.0,0.0 st=0.2,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0",
    "maxf=3 entry=0 overflow=2 ghost=0.9 gentry=1 gfreq=1 asmall=0 amain=2 st=0.05,0,0,1,0,-1,0,0.0,0.0 st=0.12,1,1,2,0,-1,1,0.0,0.0 st=0,1,1,2,2,-1,0,1.0,0.0",
    "maxf=2 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.02,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0",
    "maxf=3 entry=0 overflow=2 ghost=0.9 gentry=1 gfreq=0 asmall=0 amain=2 st=0.3,1,1,1,0,-1,1,0.0,0.0 st=0.15,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0",
    "maxf=3 entry=0 overflow=2 ghost=0.9 gentry=1 gfreq=0 asmall=0 amain=2 st=0.02,1,1,2,0,-1,1,1.0,0.0 st=0.15,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0",
    "maxf=7 entry=0 overflow=2 ghost=0.9 gentry=2 gfreq=0 asmall=0 amain=2 st=0.05,0,0,1,0,-1,0,0.0,0.0 st=0.03,1,2,2,0,-1,1,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0",
    "maxf=7 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 asmall=0 amain=1 st=0.01,1,2,1,0,-1,1,0.0,0.0 st=0,1,2,1,2,-1,0,0.0,0.0",
    "maxf=7 entry=0 overflow=2 ghost=0.5 gentry=2 gfreq=1 st=0.01,1,2,1,0,1,0 st=0.1,1,1,2,0,-1,1 st=0,1,1,2,2,-1,1",
    "maxf=1 entry=0 overflow=2 ghost=0.5 gentry=2 gfreq=0 asmall=0 amain=2 st=0.05,0,0,1,0,-1,0,0.0,0.0 st=0.15,1,1,2,0,-1,1,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0",
    "maxf=2 entry=0 overflow=3 ghost=0.9 gentry=3 gfreq=1 st=0.1,1,2,1,0,-1,1 st=0.05,1,1,3,0,-1,1 st=0.6,1,1,2,0,3,0 st=0,1,1,2,0,-1,0",
    "maxf=3 entry=0 overflow=2 ghost=2.0 gentry=1 gfreq=0 asmall=0 amain=2 st=0.3,1,2,1,0,-1,1,0.0,0.0 st=0.3,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0",
    "maxf=2 entry=0 overflow=2 ghost=0.9 gentry=0 gfreq=0 st=0.2,1,2,2,1,1,0 st=0.05,0,0,2,1,-1,0 st=0,1,1,2,2,-1,0",
    "maxf=5 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 st=0.1,1,3,1,0,-1,1 st=0,1,1,1,2,-1,0",
    "maxf=3 entry=0 overflow=2 ghost=0.9 gentry=2 gfreq=0 st=0.0125,0,0,1,1,-1,0 st=0.0375,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0",
    "maxf=3 entry=0 overflow=2 ghost=0.9 gentry=2 gfreq=0 st=0.05,0,0,1,1,-1,0 st=0.15,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0",
};
#endif
