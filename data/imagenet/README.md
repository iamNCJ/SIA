# ImageNet Data Module

## Structure

Place ImageNet Data to `data` directory, each subdirectory contains pictures in the same class.

```text
- __init__.py
- data_module.py
- class_index.json
- data
  - abacus
  - abaya
  - academic_gown
  - ...
```

`class_index.json` from https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
