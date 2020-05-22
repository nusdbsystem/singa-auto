#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from functools import wraps, reduce
from flask import request


class ParameterError(Exception):
    pass


def parser_requests():
    params = dict()
    if request.get_json():
        params['json'] = request.get_json() or {}
    if request.files.to_dict():
        params['files'] = request.files.to_dict()
    if request.form.to_dict():
        params['data'] = request.form.to_dict()
    if request.args:
        params['params'] = {k: v for k, v in request.args.items()}

    return params


def param_check(required_parameters=None):

    def decorator(f):

        @wraps(f)
        def wrapped(*args, **kwargs):
            params = parser_requests()
            if required_parameters:
                # check file fields
                for location in required_parameters:
                    for field_name in required_parameters[location]:
                        if required_parameters[location][field_name]:
                            if location not in params:
                                raise ParameterError(
                                    '{} must be provided in {}'.format(
                                        field_name, location))
                            if field_name not in params[location]:
                                raise ParameterError(
                                    '{} must be provided'.format(field_name))

            combined_params = reduce(lambda d1, d2: dict(d1, **d2),
                                     list(params.values()), {})

            return f(params=combined_params, *args, **kwargs)

        return wrapped

    return decorator

