/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.singa.auto

object Config {

  // spark table view
  val LogViewTable = "logs"
  // model
//  val SPARK_APP_NAME = "logs_process"
//  val RUNNING_MODEL="local[*]"

  // kafka config
  val TOPIC = "sa-logs"
  val BROKER = "singa-auto-kafka:9092"
  val CONSUME_GROUP = "log_consumer";

  // es config
  val ES_HOST = "elasticsearch";
  val ES_PORT = "9200";
  val ES_INDEX = "spark/logs"

  // spark stream time between 2 pull
  val TIME = "2"

  // don't store below log message at es
  val FilterMsg:Array[String] = Array("kafka.conn INFO", "kafka.consumer.subscription_state INFO")



}
