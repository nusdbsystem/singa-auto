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

package com.singa.auto.common.udf

import com.singa.auto.Config
import org.apache.spark.sql.SparkSession

class SparkUdfs(spark: SparkSession)  extends Serializable {

  def regist(): SparkSession = {

    spark.udf.register("parserMsg",
      (message: String)=>
            {
              message
            }
    )

    spark.udf.register("parserName",
      (path: String)=>
      {
        path.split("/").last
      }
    )

    spark.udf.register("filterMsg",
      (message: String) => {
        var isvalid: String = "valid";
        for (msg <- Config.FilterMsg) {
          if (message.contains(msg)) {
            isvalid = "invalid"
          }
        }
        isvalid
      }
    )

    spark

  }
}
