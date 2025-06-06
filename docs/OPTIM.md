# SKELify

Given the 2D keypoints position, SKELify provides the SKEL parameters (as well as the camera translation) that fit the 2D projections.

For SKELify without HSMR prediciton initialization, we use mean parameters as the initialization.

Before running the SKELify demo, please follow the instructions [here](./SETUP.md#eval-ds) to download the evaluation data for H36M.

> What we need for SKELify is the 2D keypoints position. But we also require image patch for visualization. If you don't want to download the image for SKELify demo, you can simply hack the code with a black image patch.

```shell
python exp/run_skelify.py
```

The results will be logged under `data_outputs/exp/SKELify-Full-demo/tb_logs`, you can view the loss curve and the visualization with TensorBoard.

```shell
tensorboard --logdir data_outputs/exp/ --port 6006
```
