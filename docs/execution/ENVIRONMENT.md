# Observed portable environment

The CPU environment was installed in a new isolated Python 3.13.9 environment on macOS arm64 on October 20, 2026. `requirements/cpu.lock.txt` contains the installed package pins, including NumPy 2.1.3, pandas 2.2.3, SciPy 1.17.1, scikit-learn 1.6.1, pydicom 3.0.2, pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.3.0 and python-gdcm 3.2.2. The final clean-source build, isolated installed-module smoke and 54 CPU tests passed. These include queue admission/STOP, header disconnection/resume, cache identity, fabricated-data CLI integration and three targeted acquisition paths. No global environment was upgraded.

The final test runner wrote its summary to an older receipt filename. The execution queue therefore correctly recorded a missing-output failure despite all three subprocesses exiting zero. The corrected V3 receipt explicitly records that post-execution correction; original logs, JUnit results and failed launcher receipt remain. This does not convert the launcher failure into a successful original receipt.

An initial clean install with SciPy 1.15.3 failed on loading its PROPACK extension under this macOS version (`__thread_bss` section file-offset error). That failed smoke remains in private evidence. Replacing only the new environment's SciPy pin with 1.17.1 allowed the rank-statistic smoke and tests to pass. Historical results and their original environments were not recomputed or relabeled as 1.17.1 results.

`requirements/inference.lock.txt` records the existing working torch 2.10.0 / torchvision 0.25.0 versions plus the tested CPU dependencies. A fresh torch installation was not tested in this session. The public synthetic smoke does not load a neural model. This distinguishes portable interface testing from a new clinical inference run.

Four fixed signed/unsigned 12-/16-bit synthetic images decoded identically through pylibjpeg and GDCM. A prospectively fixed 16-file outcome-blind source diagnostic produced 12 compressed-image exact pixel/HU agreements, four uncompressed skips and no mismatches. This limited diagnostic is not evidence that every external series is clinically compatible or every compressed source is valid.
