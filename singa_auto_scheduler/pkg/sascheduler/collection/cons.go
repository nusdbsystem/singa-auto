package collection

import framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"

const (
	AppName           = "sascheduler"
	NodeSelectorLabel = "kubernetes.io/hostname"
)

var Sum = []string{"FreeMemorySum", "MaxFreeMemory", "Number"}

type Data struct {
	Value  int64
	SValue string
}

func (s *Data) Clone() framework.StateData {
	c := &Data{
		Value: s.Value,
	}
	return c
}
