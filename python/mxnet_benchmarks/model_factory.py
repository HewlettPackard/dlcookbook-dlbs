# (c) Copyright [2017] Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model factory that creates MXNet models.

The :py:class:`ModelFactory` class scans './models' folder for classes that define
static member variable `implements`. This variable (string or a list of strings)
contains model ids that class creates.

To list supported models, run the following code:

.. code-block:: python

    from mxnet_benchmarks.model_factory import ModelFactory
    print(ModelFactory.models.keys())
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
import importlib
import inspect


def import_models():
    """Scans **./models** folder and imports models.

    See this stackoverflow thread for implementation details:
    https://stackoverflow.com/questions/3178285/list-classes-in-directory-python

    Returns:
        Dictionary that maps model id to its class.
    """
    models = {}
    fnames = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', '*.py'))
    for fname in fnames:
        mname = os.path.splitext(os.path.basename(fname))[0]
        if mname.startswith('__'):
            continue
        module = importlib.import_module("mxnet_benchmarks.models." + mname)
        for item in dir(module):
            model_cls = getattr(module, item)
            if not model_cls or not inspect.isclass(model_cls) or not hasattr(model_cls, 'implements'):
                continue	    
            implements = getattr(model_cls, 'implements')
            # Quick fix to make it compatible with Python3. Needs to be re-written.
            # if isinstance(implements, basestring):
            if isinstance(implements, ("".__class__, u"".__class__)):
                implements = [implements]
            if isinstance(implements, list) is False:
                raise ValueError("The 'implements' static member must be either a string or a list of strings"
                                 "Error in %s:%s class definition" % (fname, model_cls.__name__))
            for model_id in implements:
                # Quick fix to make it compatible with Python3. Needs to be re-written.
                # assert isinstance(model_id, basestring),\
                #     "The 'implements' static member must be either a string or a list of strings"\
                #     "Error in %s:%s class definition" % (fname, model_cls.__name__)
                if isinstance(model_id, ("".__class__, u"".__class__)) is False:
                    raise ValueError("The 'implements' static member must be either a string or a list of strings"
                                     "Error in %s:%s class definition" % (fname, model_cls.__name__))
                if model_id in models:
                    raise ValueError("Model %s implements same model as %s (%s)" % (model_cls.__name__,
                                                                                    models[model_id].__name__,
                                                                                    model_id))
                models[model_id] = model_cls
    return models


class ModelFactory(object):
    """MXNet model factory that creates models."""

    models = import_models()

    @staticmethod
    def get_model(params):
        """Return model identified by *params['model']*.

        Args:
            params (dict): Parameters of the model. Must include at least *model* key that identifies model id.

        Returns:
            Model instance.
        """
        model = params.get('model', None)
        if model is None:
            raise ValueError("No model name found in params")
        if model not in ModelFactory.models:
            raise ValueError("Unsupported model: '%s'" % model)
        return ModelFactory.models[model](params)
