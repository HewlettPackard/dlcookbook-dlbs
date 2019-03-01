# TensorFlow for CPUs with MKL support

- `GitHub hash tag`: [d9d5ab289d08da9b3fa0a4feccf8dfa87a7b669e](https://github.com/tensorflow/tensorflow/commit/d9d5ab289d08da9b3fa0a4feccf8dfa87a7b669e).
- `Date`: 02.27.2019
- `Params`:
  ```python
  import tensorflow as tf
  from tensorflow.python.framework import test_util

  print(tf.__version__)            # 1.12.0
  print(test_util.IsMklEnabled())  # True
  ```
