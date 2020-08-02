package score

import (
	"github.com/naili-xing/singa_auto_scheduler/pkg/sascheduler/collection"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

var Weights = make(map[string]int64)

func Sum2WeightMap() {
	var weight int64 = 1
	for _, ele := range collection.Sum {
		// by default , all weight is 1 for each key in Sum: filter.Sum
		Weights[ele] = weight
	}
}

func CalculateCollectScore(
	state *framework.CycleState,
	node *nodeinfo.NodeInfo) (int64, error) {

	Sum2WeightMap()
	var score int64 = 0
	for v, w := range Weights {
		s, err := CalculateValueScore(v, state, node)
		if err != nil {
			return 0, err
		}
		score += s * w
	}
	return score, nil
}

func CalculateValueScore(
	value string,
	state *framework.CycleState,
	node *nodeinfo.NodeInfo) (int64, error) {

	d, err := state.Read(framework.StateKey("Max" + value))
	if err != nil {
		klog.V(3).Infof("Error Get CycleState Info: %v", err)
		return 0, err
	}
	// type assertion
	return collection.StrToInt64(node.Node().Labels["samonitor/"+value]) * 100 / d.(*collection.Data).Value, nil
}
