# PyFrame documentation

Reference documentation for PyFrame. This folder grows over time; add new pages
here and link them from the list below.

> **Documentation home:** [eden.report/docs](https://www.eden.report/docs) hosts the fullest version of these docs, including a short annotated live diagram of the pipeline. The pages in this folder are the in-repo reference mirror.

## Contents

- [Output reference](output.md) - every field in the JSON / `ScanResult`, explained.
- [Performance](performance.md) - measured throughput, per-stage timing, capacity sizing, and a results log.

## See also

- [eden.report/docs](https://www.eden.report/docs) - the documentation website: expanded guides and an annotated live pipeline diagram.
- Project [README](../README.md) - install, quickstart, CLI, and the pipeline diagram.

## References

The temporal sampling and windowing design (uniform-by-time coverage as a recall floor,
motion as a content-blind cost lever, and grouping flagged frames into temporal windows)
is informed by the temporal action segmentation survey:

> Guodong Ding, Fadime Sener, and Angela Yao. "Temporal Action Segmentation: An Analysis of Modern Techniques." arXiv:2210.10352, 2023. https://arxiv.org/abs/2210.10352

```bibtex
@misc{ding2023temporalactionsegmentationanalysis,
      title={Temporal Action Segmentation: An Analysis of Modern Techniques},
      author={Guodong Ding and Fadime Sener and Angela Yao},
      year={2023},
      eprint={2210.10352},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2210.10352},
}
```
