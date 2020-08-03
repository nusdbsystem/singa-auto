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
FROM logstash:7.7.0

MAINTAINER NailiXing <xingnaili14@gmail.com>


RUN /usr/share/logstash/bin/logstash-plugin install logstash-input-http_poller
RUN /usr/share/logstash/bin/logstash-plugin install logstash-input-exec
RUN /usr/share/logstash/bin/logstash-plugin install logstash-filter-json_encode


EXPOSE 9600 5044

ARG LOGSTASH_DOCKER_WORKDIR_PATH

WORKDIR $LOGSTASH_DOCKER_WORKDIR_PATH

COPY log_minitor/config/logstash.conf $LOGSTASH_DOCKER_WORKDIR_PATH/logstash.conf

CMD bin/logstash -f logstash.conf
