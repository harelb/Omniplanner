from types import SimpleNamespace
import numpy as np
import pytest
from dsg_pddl.pddl_grounding import PddlSymbol
from dsg_pddl.dsg_pddl_planning import parameterize_object_approach, parameterize_pick_object, parameterize_goto_poi


def test_pick_uses_observed_object_height_not_its_place():
    obj = PddlSymbol('o1','object',[],np.array([1.,0.]),np.array([1.,0.,.82]))
    place = PddlSymbol('t1','place',[],np.array([1.2,.2]))
    robot,target=parameterize_pick_object(None,{'o1':obj,'t1':place},('pick-object','o1','t1'),[0.,0.])
    assert np.allclose(target,[1.,0.,.82]) and np.allclose(robot,[0.,0.])


def test_already_near_object_stays_at_observed_base_pose_and_faces_target():
    def forbidden(*a):raise AssertionError('No route to object center should be requested')
    planner=SimpleNamespace(get_external_path=forbidden)
    route,last=parameterize_object_approach(planner,[0.,0.],[0.,.9],(.65,1.05))
    assert np.allclose(route,[[0.,0.,np.pi/2],[0.,0.,np.pi/2]])
    assert np.allclose(last,[0.,0.])


def test_far_object_approach_uses_reachable_observed_place():
    planner=SimpleNamespace(node_positions=np.array([[3.,0.],[2.1,0.],[2.3,.1]]),
        get_external_distance=lambda a,b:np.linalg.norm(np.array(a)-b),
        get_external_path=lambda a,b:[a,b])
    route,last=parameterize_object_approach(planner,[0.,0.],[3.,0.],(.65,1.05))
    assert np.allclose(last,[2.1,0.])
    assert np.allclose(route[-1],[2.1,0.,0.])
    with pytest.raises(ValueError,match='No observed'):
        parameterize_object_approach(planner,[0.,0.],[9.,0.],(.65,1.05))


def test_reached_stance_preserves_observed_heading_for_arm_gaze():
    def forbidden(*a):raise AssertionError('Already in pickup range')
    planner=SimpleNamespace(get_external_path=forbidden)
    route,last=parameterize_object_approach(planner,[0.,0.],[0.,.9],(.65,1.05),-.4)
    assert np.allclose(route,[[0.,0.,-.4],[0.,0.,-.4]])
    assert np.allclose(last,[0.,0.])
    with pytest.raises(ValueError,match='heading'):
        parameterize_object_approach(planner,[0.,0.],[0.,.9],(.65,1.05),float('nan'))


def test_travel_to_new_stance_does_not_reuse_old_heading():
    planner=SimpleNamespace(node_positions=np.array([[2.1,0.]]),
        get_external_distance=lambda a,b:float(np.linalg.norm(np.array(a)-b)),
        get_external_path=lambda a,b:[a,b])
    route,_=parameterize_object_approach(planner,[0.,0.],[3.,0.],(.65,1.05),-.4)
    assert np.allclose(route[-1],[2.1,0.,0.])


def test_following_route_starts_at_actual_previous_standoff():
    planner=SimpleNamespace(get_external_path=lambda a,b:[a,b])
    symbols={'o1':PddlSymbol('o1','object',[],np.array([3.,0.])),
        't1':PddlSymbol('t1','place',[],np.array([5.,0.]))}
    route,_=parameterize_goto_poi(planner,symbols,('goto-poi','o1','t1'),last_pose=[2.1,0.])
    assert np.allclose(route[0],[2.1,0.])


@pytest.mark.parametrize('status,expected',[('admitted','mug'),('candidate','UNKNOWN')])
def test_context_preserves_only_admitted_open_set_semantics(status,expected):
    from omniplanner.omniplanner import DsgContextProvider
    attrs=SimpleNamespace(position=np.array([1.,0.,.82]),metadata=SimpleNamespace(
        get=lambda:{'admitted_semantics':{'admission_status':status,'class':'mug'}}))
    node=SimpleNamespace(attributes=attrs,layer=SimpleNamespace(layer=2,partition=0))
    graph=SimpleNamespace(find_node=lambda ns:node,
        get_labelspace=lambda *a:SimpleNamespace(get_node_category=lambda node:'UNKNOWN'))
    assert DsgContextProvider(graph)['o1']['semantic_label']==expected
