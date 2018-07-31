"""This file defines a class that loads arbitrary ONNX models.

Current limitations:
  1. Only single precision support (float32).
  2. Only inference phase.
  3. Models I tested worked only with batch specified in metadata. The error
     seems to be related with 'reshape' operator that in my experimetns
     reshapes input tensor to (1,-1) where first dimension is a batch dimension
     and the second dimension is a feature dim, not fixed in this case. I think
     it should be quite the opposite: (-1,N) wheren batch dimension is not fixed
     and N - number of features - is computed statically at export phase.
  4. Model title is the model file name without extension unless a model metadata
     contains 'name' key with string value.
"""
from __future__ import absolute_import
from __future__ import print_function
import os
from mxnet_benchmarks.models.model import Model
import mxnet.contrib.onnx as onnx_mxnet


class _ONNXModel(Model):
    """ Class to load arbitrary ONNX models."""

    implements = '_onnx_model'

    @property
    def output(self):
        return self.__output

    def __init__(self, params):
        #
        if 'model_url' not in params:
            self.__output = None
            print("[ERROR] No 'model_url' found in params.")
            return
        # Load model's metadata and check it has minimal set of required parameters.
        metadata = onnx_mxnet.get_model_metadata(params['model_url'])
        shapes = [None]*2    # Shapes for [input, output] excluding batch size
        names = {}           # Rename ONNX symbols to expectable names.
        standard_names = [   # This is what DLBS expects (standard symbol names)
            'data',          #   - Name of an input symbol.
            'softmax_label'  #   - Name of an output symbol in training phase. In
                             #     inference phase this name does not matter.
        ]
        print("ONNX metadata: %s" % str(metadata))
        for i, mkey in enumerate(['input_tensor_data', 'output_tensor_data']):
            if mkey not in metadata:
                self.__output = None
                raise RuntimeError("Invalid ONNX model (no '%s' found in metadata." % mkey)
            shapes[i] = metadata[mkey]
            if isinstance(shapes[i], list) and len(shapes[i]) > 1:
                print("[ERROR] ONNX metadata defines > 1 (%d) %s tensors" % (len(shapes[i]), mkey))
            shapes[i] = shapes[i][0] if isinstance(shapes[i], list) else shapes[i]
            if isinstance(shapes[i], tuple):
                names[shapes[i][0]] = standard_names[i]
                shapes[i] = shapes[i][1]
        if 'name' in metadata:
            model_name = str(metadata['name'])
        else:
            model_name = os.path.splitext(os.path.basename(params['model_url']))[0]
        #
        Model.check_parameters(
            params,
            {'name': model_name, 'input_shape': shapes[0][1:], 'num_classes': shapes[1][1],
             'phase': 'training', 'dtype': 'float32',
             'model_opts': {}}
        )
        # shapes[0][0] should be the batch size, if it's <=0, this should mean batch size is
        # not fixed.
        if shapes[0][0] > 0:
            if 'batch_size' in params['model_opts'] and params['model_opts']['batch_size'] != shapes[0][0]:
                print(
                    "[WARNING] Batch size in model options (%d) is different from the one loaded from ONNX"\
                    "file (%d). I will use ONNX value and overwrite the one provided by a user."\
                    % (params['model_opts']['batch_size'], shapes[0][0])
                )
            params['model_opts']['batch_size'] = shapes[0][0]
        #
        print("Models parameters: %s" % str(params))
        Model.__init__(self, params)
        self.__output, arg_params, aux_params = onnx_mxnet.import_model(params['model_url'])
        self._init_params = (arg_params, aux_params)
        # Rename input/output symbols
        for sym in self.__output.get_internals():
            for cur_name, new_name in names.iteritems():
                if sym.name == cur_name:
                    print("Renaming symbols: %s -> %s" % (cur_name, new_name))
                    sym._set_attr(name=new_name)
        #print(self.__output.list_inputs())
