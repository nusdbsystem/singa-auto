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
// under the License.\
package collection

import (
	"github.com/NVIDIA/gpu-monitoring-tools/bindings/go/nvml"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/log"
)

type GPU struct {
	ID         uint
	Health     string
	FreeMemory uint64 // free memory of a GPU
	Device     nvml.Device
}

type GpuScore struct {
	Device GPU
	Score  uint64
}

// check if GPU is healthy
func CheckHealth() string {
	for _, g := range GPUs {
		// if one is healthy, then return
		if g.Health == "Healthy" {
			return "Healthy"
		}
	}
	return "Unhealthy"
}

// check if there is GPU
func CheckGPU() bool {
	err := nvml.Init()
	if err != nil {
		log.ErrPrint("CheckGPU", err)
	}
	defer func() {
		if err := nvml.Shutdown(); err != nil {
			log.ErrPrint("CheckGPU", err)
		}
	}()
	count, err := nvml.GetDeviceCount()
	if err != nil {
		log.ErrPrint("CheckGPU", err)
	}
	// check
	if count > 0 {
		return true
	}
	return false
}

// count number of GPU
func CountGPU() uint {
	err := nvml.Init()
	if err != nil {
		log.ErrPrint("CountGPU", err)
	}
	defer func() {
		if err := nvml.Shutdown(); err != nil {
			log.ErrPrint("CountGPU", err)
		}
	}()
	count, err := nvml.GetDeviceCount()
	if err != nil {
		log.ErrPrint("CountGPU", err)
	}
	return count
}

func AddGPUInfo() error {
	health := "Healthy"
	err := nvml.Init()
	if err != nil {
		log.ErrPrint("AddGPUInfo", err)
		return err
	}
	// define def below the error check
	// because if there is error, no need to call def, return directly
	// if error is not nil, it may cause panic when release resource
	defer func() {
		if err := nvml.Shutdown(); err != nil {
			log.ErrPrint("AddGPUInfo", err)
		}
	}()
	count, err := nvml.GetDeviceCount()
	if err != nil {
		log.ErrPrint("AddGPUInfo", err)
		return err
	}
	for i := uint(0); i < count; i++ {
		device, err := nvml.NewDevice(i)
		if err != nil {
			log.ErrPrint("AddGPUInfo", err)
		}
		status, err := device.Status()
		if err != nil {
			log.ErrPrint("AddGPUInfo", err)
			health = "Unhealthy"
		}

		// add gpu info to global gpu list
		GPUs = append(GPUs, GPU{
			ID:         i,
			Health:     health,
			FreeMemory: *status.Memory.Global.Free,
			Device:     *device,
		})

	}

	return err
}

func UpdateGPU() error {
	var NewGPUs []GPU
	var health string = "Healthy"
	err := nvml.Init()
	if err != nil {
		log.ErrPrint("UpdateGPU", err)
	}
	defer func() {
		if err := nvml.Shutdown(); err != nil {
			log.ErrPrint("UpdateGPU", err)
		}
	}()
	count, err := nvml.GetDeviceCount()
	if err != nil {
		log.ErrPrint("UpdateGPU", err)
	}
	for i := uint(0); i < count; i++ {
		device, err := nvml.NewDevice(i)
		if err != nil {
			log.ErrPrint("UpdateGPU", err)
		}
		status, err := device.Status()
		if err != nil {
			log.ErrPrint("UpdateGPU", err)
			health = "Unhealthy"
		}
		NewGPUs = append(NewGPUs, GPU{
			ID:         i,
			Health:     health,
			FreeMemory: *status.Memory.Global.Free,
			Device:     *device,
		})
	}

	// update global gpu
	GPUs = NewGPUs
	return err
}
