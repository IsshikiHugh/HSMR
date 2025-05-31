# Miscellaneous Tools

## Requirements

```shell
pip install tyro  # for fast CLI application building
```

## Export Meshes

Once you get the output file `*.npz` from demo, you can export the meshes using the following script.

```python
python exp/misc/export_mesh.py \
    --input_path "data_outputs/demos/xxx.npz" \
    --outputs_root "data_outputs/demos/xxx"
```



