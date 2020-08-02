package sascheduler

import (
	"context"
	"fmt"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sascheduler/collection"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sascheduler/filter"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sascheduler/score"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sascheduler/sort"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

var (
	_ framework.QueueSortPlugin  = &SingaAutoScheduler{}
	_ framework.FilterPlugin     = &SingaAutoScheduler{}
	_ framework.PostFilterPlugin = &SingaAutoScheduler{}
	_ framework.ScorePlugin      = &SingaAutoScheduler{}
	_ framework.ScoreExtensions  = &SingaAutoScheduler{}
)

type Args struct {
	KubeConfig string `json:"kubeconfig,omitempty"`
	Master     string `json:"master,omitempty"`
}

type SingaAutoScheduler struct {
	args   *Args
	handle framework.FrameworkHandle
}

func (s *SingaAutoScheduler) Name() string {
	return collection.AppName
}

// This is pre-filter method, this is for v1.18+
func (s *SingaAutoScheduler) PreFilter(
	ctx context.Context,
	state *framework.CycleState,
	p *v1.Pod) *framework.Status {

	klog.V(3).Infof("pre-filter pod: %v", p.Name)
	return framework.NewStatus(framework.Success, "")
}

// called on each node
func (s *SingaAutoScheduler) Filter(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	node *schedulernodeinfo.NodeInfo) *framework.Status {

	klog.V(3).Infof("filter pod: %v, node: %v", pod.Name, node.Node().Name)

	// if the pod has pre defined node info
	if nodeName, ok := pod.Spec.NodeSelector[collection.NodeSelectorLabel]; ok {
		klog.V(3).Infof("filter: pod NodeSelector is: %v", pod.Spec.NodeSelector)

		// if current node is "the node", return success
		if node.Node().Labels[collection.NodeSelectorLabel] == nodeName {
			return framework.NewStatus(framework.Success, "")
		} else {
			// if current node is not "the node", return Unscheduable
			return framework.NewStatus(framework.Unschedulable,
				"Node:"+node.Node().Name+" Pod is assigned to "+nodeName)
		}
	} else {
		// check if node has gpu, and if gpu is available
		if ok, msg := filter.CheckGPUStatus(node); ok {

			// init podInfo, type of pointer
			var podInfo = &filter.PodInfo{Pod: pod, Node: node}

			if !podInfo.PodFitsMemory() {
				return framework.NewStatus(
					framework.Unschedulable,
					"Node:"+node.Node().Name+" GPU Memory Not Fit")
			}
			if !podInfo.PodFitsNumber() {
				return framework.NewStatus(
					framework.Unschedulable,
					"Node:"+node.Node().Name+" GPU Memory Not Fit")
			}
			return framework.NewStatus(
				framework.Success,
				"")
		} else {
			return framework.NewStatus(
				framework.Unschedulable,
				"Node: "+node.Node().Name+msg)
		}
	}
}

// this is used to sort the pods according to the priority
func (s *SingaAutoScheduler) Less(podInfo1, podInfo2 *framework.PodInfo) bool {
	return sort.Less(podInfo1, podInfo2)
}

// post-filter in 1.17 is also called pre-score in 1.18,
// this is for v 1.17+, 1.18 doesnt have filteredNodesStatuses in in pre-score
// can not get filter result. so switched to 1.17
// all filtered node info can be found there
func (s *SingaAutoScheduler) PostFilter(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	nodes []*v1.Node,
	filteredNodesStatuses framework.NodeToStatusMap) *framework.Status {

	// state is data store shared by all plugins. here add customer info to state
	klog.V(3).Infof("collect info for scheduling  pod: %v", pod.Name)

	// if the pod dont have pre-assigned node info, find one for it
	if _, ok := pod.Spec.NodeSelector[collection.NodeSelectorLabel]; !ok {

		// select a best node and gpu according to node's GpuSummary
		filter.MatchGpu2Pod(pod, nodes, filteredNodesStatuses)

	}

	return filter.ParallelCollection(filter.Workers, state, nodes, filteredNodesStatuses)
}

// Score is called on each filtered node,
// It must return success and an integer indicating the rank of the node.
// state is data store shared by all plugins.
func (s *SingaAutoScheduler) Score(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	nodeName string) (int64, *framework.Status) {

	// This information should only be used before "Reserve" point
	// Get the node info of the filtered node
	nodeInfo, err := s.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error,
			fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	// if the pod has pre defined node info
	if nodeName, ok := pod.Spec.NodeSelector[collection.NodeSelectorLabel]; ok {
		klog.V(3).Infof("score: pod NodeSelector is: %v", pod.Spec.NodeSelector)

		// if current node is "the node", return 100
		if nodeInfo.Node().Labels[collection.NodeSelectorLabel] == nodeName {
			return 100, framework.NewStatus(framework.Error,
				fmt.Sprintf("Score Node Error: %v", err))
			// if current node is not "the node", return 0
		} else {
			return 0, framework.NewStatus(framework.Error,
				fmt.Sprintf("Score Node Error: %v", err))

		}
	} else {
		// if assign gpu or node fail, apply the default method to calculate a score for each node
		// calculate the score according to node and pod
		res, err := score.Score(state, nodeInfo)
		if err != nil {
			return 0, framework.NewStatus(framework.Error,
				fmt.Sprintf("Score Node Error: %v", err))
		}
		klog.V(3).Infof("node : %v sa-scheduler score: %v", nodeName, s)
		return res, framework.NewStatus(framework.Success, "")
	}
}

func (s *SingaAutoScheduler) NormalizeScore(
	ctx context.Context,
	state *framework.CycleState,
	p *v1.Pod,
	scores framework.NodeScoreList) *framework.Status {

	var (
		highest int64 = 0
		lowest        = scores[0].Score
	)
	for _, nodeScore := range scores {
		if nodeScore.Score < lowest {
			lowest = nodeScore.Score
		}
	}
	if lowest < 0 {
		for i := range scores {
			scores[i].Score -= lowest
		}
	}
	for _, nodeScore := range scores {
		if nodeScore.Score > highest {
			highest = nodeScore.Score
		}
	}
	// Set Range to [0-100]
	for i, nodeScore := range scores {
		scores[i].Score = nodeScore.Score * framework.MaxNodeScore / highest
	}
	return framework.NewStatus(framework.Success, "")
}

func (s *SingaAutoScheduler) ScoreExtensions() framework.ScoreExtensions {
	return s
}

func New(
	configuration *runtime.Unknown,
	f framework.FrameworkHandle) (framework.Plugin, error) {

	args := &Args{}
	if err := framework.DecodeInto(configuration, args); err != nil {
		return nil, err
	}
	klog.V(3).Infof("get plugin config args: %+v", args)

	return &SingaAutoScheduler{
		args:   args,
		handle: f,
	}, nil
}
