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
	"errors"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/log"
	"strconv"
	"strings"
)

func (g *GPU) CalculateScoreWithMaxFreeMemoryMode() uint64 {
	// with the simplest method, score is the value of FreeMemory
	return g.FreeMemory
}

// get the summary of GPU, with format of GPU_0.GPU_1.GPU_2 for example
func CalculateGpuSummary() string {
	var GpuFreeMemorys []string
	for _, g := range GPUs {
		element := strings.Join(
			[]string{
				strconv.FormatInt(int64(g.ID), 10),
				strconv.FormatInt(int64(g.FreeMemory), 10)},
			"_")
		GpuFreeMemorys = append(GpuFreeMemorys, element)
	}
	return strings.Join(GpuFreeMemorys, ".")
}

// calculate the total GPU free memory inside a node
func CalculateGPUFreeMemorySum() uint64 {
	var Sum uint64 = 0
	for _, g := range GPUs {
		Sum += g.FreeMemory
	}
	if Sum == 0 {
		log.ErrPrint("CalculateGPUFreeMemorySum", errors.New("The Sum of the GPU memory is 0. "))
		return 0
	}
	return Sum
}

func CalculateBestGPUWithMaxFreeMemoryMode() GPU {
	var (
		GpuScores []GpuScore
		Max       uint64 = 0
		Device    GPU
	)
	for _, g := range GPUs {
		if g.Health == "Unhealthy" {
			GpuScores = append(GpuScores, GpuScore{
				Device: g,
				Score:  0,
			})
			continue
		} else {
			GpuScores = append(GpuScores, GpuScore{
				Device: g,
				Score:  g.CalculateScoreWithMaxFreeMemoryMode(),
			})
		}
	}
	for _, s := range GpuScores {
		if s.Score > Max {
			Max = s.Score
			Device = s.Device
		}
	}
	return Device
}
