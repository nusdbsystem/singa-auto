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
package collection

import (
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/log"
	"reflect"
	"strconv"
	"strings"
)

type NodeInfo struct {
	Gpu           bool   // If the node has GPU, the value is true.
	Health        string // If all GPU is unhealthy,the value is unhealthy. if one is healthy, it;s healthy
	FreeMemorySum uint64 // The Sum of the free memory of the GPUs in one node
	MaxFreeMemory uint64 // Max freeMemory of a gpu in one node
	GpuSummary    string // "Gpu0_freeMemory" examples: Gpu0_100.Gpu1_200.Gpu2_300
	Number        uint   // Number of GPus
}

func InitNodeInfoWithMaxFreeMemoryMode() {
	InitModelNodeInfo(CalculateBestGPUWithMaxFreeMemoryMode)
}

func UpdateNodeInfoWithMaxFreeMemoryMode() {
	UpdateModelNodeInfo(CalculateBestGPUWithMaxFreeMemoryMode)
}

func InitModelNodeInfo(Mode func() GPU) {
	// Generate a global gpu list
	if err := AddGPUInfo(); err != nil {
		log.ErrPrint("InitModelNodeInfo", err)
	}
	// get the most greatest device to show, greatest is defined by the model,
	// currently, we measure the score by using free memory, bigger is better
	Device := Mode()
	Nodeinfo = NodeInfo{
		Gpu:           CheckGPU(),
		Health:        CheckHealth(),
		FreeMemorySum: CalculateGPUFreeMemorySum(),
		MaxFreeMemory: Device.FreeMemory,
		GpuSummary:    CalculateGpuSummary(),
		Number:        CountGPU(),
	}
}

func UpdateModelNodeInfo(Mode func() GPU) {
	if err := UpdateGPU(); err != nil {
		log.ErrPrint("UpdateModelNodeInfo", err)
	}
	Device := Mode()
	Nodeinfo = NodeInfo{
		Gpu:           CheckGPU(),
		Health:        CheckHealth(),
		FreeMemorySum: CalculateGPUFreeMemorySum(),
		MaxFreeMemory: Device.FreeMemory,
		GpuSummary:    CalculateGpuSummary(),
		Number:        CountGPU(),
	}
}

func NodeInfoToMap() map[string]string {
	m := make(map[string]string)
	var elem reflect.Value = reflect.ValueOf(Nodeinfo)
	var relType reflect.Type = reflect.TypeOf(Nodeinfo)
	//relType := elem.Type()
	for i := 0; i < relType.NumField(); i++ {
		if elem.Field(i).Type() == reflect.TypeOf("") {
			m["samonitor/"+relType.Field(i).Name] = strings.Replace(elem.Field(i).String(), " ", "-", -1)
		} else if elem.Field(i).Type() == reflect.TypeOf(true) {
			m["samonitor/"+relType.Field(i).Name] = BoolToString(elem.Field(i).Bool())
		} else {
			m["samonitor/"+relType.Field(i).Name] = strconv.Itoa(int(elem.Field(i).Uint()))
		}
	}
	return m
}

func BoolToString(value bool) string {
	if value {
		return "True"
	} else {
		return "False"
	}
}
