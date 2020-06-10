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
package com.singa.auto.monitor.processer

import com.singa.auto.Config
import org.apache.spark.sql.catalyst.util.StringUtils
import org.apache.spark.sql.{DataFrame, SparkSession}

class LogEvents(spark: SparkSession, dataFrame: DataFrame) extends BaseProcess {

  override def sparkSQl(dataFrame: DataFrame) :String = {

    var sqlBuffer = new StringBuffer()

    sqlBuffer.append("SELECT ")
    sqlBuffer.append(FiledsParse.getFileName(dataFrame)).append(",")
    sqlBuffer.append(FiledsParse.getMessage(dataFrame))
    sqlBuffer.append(" FROM ")
    sqlBuffer.append(tempView(LogEvents.TABLE))
    sqlBuffer.append(" WHERE ")
    sqlBuffer.append(FiledsParse.filterMessage(dataFrame))

    println("Using SQL",sqlBuffer.toString)

    sqlBuffer.toString
    }
}


object LogEvents {
  val TABLE:String = Config.LogViewTable;

  def stream(spark: SparkSession, dataFrame: DataFrame): Unit = {
    println("-------------stream-Events-------------")
    val app = new LogEvents(spark, dataFrame);
    app.process(spark, dataFrame,TABLE, null)

  }
}

