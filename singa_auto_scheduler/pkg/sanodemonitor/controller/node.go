//
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
package controller

import (
	"encoding/json"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/collection"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/log"
)

func InitNodeInfo(mode string) {
	log.Print("Init Monitor, mode:" + mode)
	switch mode {
	case "MaxFreeMemory":
		collection.InitNodeInfoWithMaxFreeMemoryMode()
	}
}

func UpdateNodeInfo(mode string) {
	switch mode {
	case "MaxFreeMemory":
		collection.UpdateNodeInfoWithMaxFreeMemoryMode()
	}
}

func PrintNodeInfo() {
	s, err := json.Marshal(&collection.Nodeinfo)
	if err != nil {
		log.ErrPrint("PrintNodeInfo", err)
	}
	log.Print("Node Info: " + string(s))
}
