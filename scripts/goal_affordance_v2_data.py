"""Build the bilingual, conversational Goal-Affordance v2 candidate pool.

V2 is intentionally narrower than v1. It targets short prompts where a salient
distance, speed, or immediate-start cue supports a locally sensible action that
fails the actual goal. Every semantic scenario has matched English and Korean
surfaces and four paired conditions.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "results" / "goal_affordance_v2_development"


@dataclass(frozen=True)
class Surface:
    semantic_id: str
    language: str
    family: str
    hostile_question: str
    explicit_question: str
    neutral_question: str
    counterfactual_question: str
    correct_action: str
    lure_action: str
    heuristic: str
    rationale: str

    @property
    def pair_id(self) -> str:
        return f"{self.semantic_id}_{self.language}"


SURFACES = (
    Surface(
        semantic_id="vehicle_car_wash",
        language="en",
        family="goal_bound_vehicle",
        hostile_question=(
            "I want to wash my car. The car wash is only 50 meters ahead, so walking "
            "would be quicker than pulling out and parking again. What should I do?"
        ),
        explicit_question=(
            "I want to wash my car. The car wash is only 50 meters ahead, so walking "
            "would be quicker. The car itself has to reach the wash bay. What should I do?"
        ),
        neutral_question=(
            "I want to wash my car at the car wash 50 meters ahead. What should I do?"
        ),
        counterfactual_question=(
            "I only want to ask the car wash about its prices, not wash the car today. "
            "It is 50 meters ahead. What should I do?"
        ),
        correct_action="drive the car there",
        lure_action="walk there",
        heuristic="short_distance_implies_walk",
        rationale="The object of the goal is the car, so the car must reach the wash bay.",
    ),
    Surface(
        semantic_id="vehicle_car_wash",
        language="ko",
        family="goal_bound_vehicle",
        hostile_question=(
            "세차를 하고 싶다. 세차장은 50m 앞이라 차를 빼고 다시 주차하는 것보다 "
            "걸어가는 편이 빠르다. 어떻게 가야 할까?"
        ),
        explicit_question=(
            "세차를 하고 싶다. 세차장은 50m 앞이라 걸어가는 편이 빠르다. 하지만 "
            "세차할 차가 세차장 안으로 들어가야 한다. 어떻게 가야 할까?"
        ),
        neutral_question="50m 앞 세차장에서 내 차를 세차하고 싶다. 어떻게 가야 할까?",
        counterfactual_question=(
            "오늘 세차는 하지 않고 50m 앞 세차장에 가격만 물어보려 한다. "
            "어떻게 가야 할까?"
        ),
        correct_action="차를 운전해 간다",
        lure_action="걸어간다",
        heuristic="short_distance_implies_walk",
        rationale="목표의 대상은 자동차이므로 자동차가 세차장에 도착해야 한다.",
    ),
    Surface(
        semantic_id="vehicle_refuel",
        language="en",
        family="goal_bound_vehicle",
        hostile_question=(
            "I need to put gas in my car. The station is 100 meters down the road, and "
            "walking there would be much faster in this traffic. What should I do?"
        ),
        explicit_question=(
            "I need to put gas in my car. The station is 100 meters away and walking is "
            "faster, but the car must be beside the pump to be fueled. What should I do?"
        ),
        neutral_question=(
            "I need to refuel my car at the station 100 meters away. What should I do?"
        ),
        counterfactual_question=(
            "I only want to read the fuel price posted at the station 100 meters away; "
            "I am not refueling now. What should I do?"
        ),
        correct_action="drive the car to the station",
        lure_action="walk to the station",
        heuristic="short_distance_implies_walk",
        rationale="Refueling requires the car to be present at the pump.",
    ),
    Surface(
        semantic_id="vehicle_refuel",
        language="ko",
        family="goal_bound_vehicle",
        hostile_question=(
            "차에 기름을 넣어야 한다. 주유소는 100m 앞이고 길이 막혀 걸어가는 편이 "
            "훨씬 빠르다. 어떻게 가야 할까?"
        ),
        explicit_question=(
            "차에 기름을 넣어야 한다. 100m 앞 주유소까지는 걸어가는 편이 빠르지만, "
            "주유하려면 차가 주유기 옆에 있어야 한다. 어떻게 가야 할까?"
        ),
        neutral_question="100m 앞 주유소에서 차에 기름을 넣어야 한다. 어떻게 가야 할까?",
        counterfactual_question=(
            "지금 주유하지 않고 100m 앞 주유소의 가격표만 보려 한다. 어떻게 가야 할까?"
        ),
        correct_action="차를 운전해 주유소로 간다",
        lure_action="주유소까지 걸어간다",
        heuristic="short_distance_implies_walk",
        rationale="주유하려면 자동차가 주유기 앞에 도착해야 한다.",
    ),
    Surface(
        semantic_id="vehicle_tire_air",
        language="en",
        family="goal_bound_vehicle",
        hostile_question=(
            "I need to add air to my car's low tire. The service station is just across "
            "the parking lot, and walking across is easier than driving around. What should I do?"
        ),
        explicit_question=(
            "I need to add air to my car's low tire. Walking to the nearby service station "
            "is easier, but the tire is attached to the car. What should I do?"
        ),
        neutral_question=(
            "I need to add air to my car's low tire at the service station across the "
            "parking lot. What should I do?"
        ),
        counterfactual_question=(
            "I only want to ask whether the service station's air pump is working; I am "
            "not filling the tire now. What should I do?"
        ),
        correct_action="drive the car to the service station",
        lure_action="walk to the service station",
        heuristic="nearby_destination_implies_walk",
        rationale="The tire that needs air is attached to the car.",
    ),
    Surface(
        semantic_id="vehicle_tire_air",
        language="ko",
        family="goal_bound_vehicle",
        hostile_question=(
            "차 타이어에 공기를 넣어야 한다. 정비소는 주차장 건너편이라 차로 돌아가는 "
            "것보다 걸어가는 편이 쉽다. 어떻게 가야 할까?"
        ),
        explicit_question=(
            "차 타이어에 공기를 넣어야 한다. 가까운 정비소까지 걸어가는 편이 쉽지만, "
            "타이어는 차에 붙어 있다. 어떻게 가야 할까?"
        ),
        neutral_question=(
            "주차장 건너편 정비소에서 차 타이어에 공기를 넣어야 한다. 어떻게 가야 할까?"
        ),
        counterfactual_question=(
            "지금 타이어에 공기를 넣지는 않고 정비소 공기 주입기가 작동하는지만 "
            "물어보려 한다. 어떻게 가야 할까?"
        ),
        correct_action="차를 운전해 정비소로 간다",
        lure_action="정비소까지 걸어간다",
        heuristic="nearby_destination_implies_walk",
        rationale="공기를 넣을 타이어가 자동차에 붙어 있다.",
    ),
    Surface(
        semantic_id="vehicle_inspection",
        language="en",
        family="goal_bound_vehicle",
        hostile_question=(
            "I want the mechanic on the next block to inspect the noise from my car. "
            "Walking there is quicker than finding a way through the one-way street. "
            "What should I do?"
        ),
        explicit_question=(
            "I want the nearby mechanic to inspect the noise from my car. Walking there "
            "is quicker, but the mechanic needs the noisy car present. What should I do?"
        ),
        neutral_question=(
            "I want the mechanic on the next block to inspect a noise from my car. "
            "What should I do?"
        ),
        counterfactual_question=(
            "I only want to ask the mechanic when the shop closes; the car is not being "
            "inspected today. What should I do?"
        ),
        correct_action="drive the car to the mechanic",
        lure_action="walk to the mechanic",
        heuristic="fastest_person_route_ignores_target",
        rationale="The mechanic must hear or inspect the car that is making the noise.",
    ),
    Surface(
        semantic_id="vehicle_inspection",
        language="ko",
        family="goal_bound_vehicle",
        hostile_question=(
            "다음 블록 정비사에게 내 차에서 나는 소리를 점검받고 싶다. 일방통행로를 "
            "돌아가는 것보다 걸어가는 편이 빠르다. 어떻게 가야 할까?"
        ),
        explicit_question=(
            "가까운 정비사에게 내 차의 소리를 점검받고 싶다. 걸어가는 편이 빠르지만 "
            "정비사가 소리 나는 차를 직접 봐야 한다. 어떻게 가야 할까?"
        ),
        neutral_question=(
            "다음 블록 정비사에게 내 차에서 나는 소리를 점검받고 싶다. "
            "어떻게 가야 할까?"
        ),
        counterfactual_question=(
            "오늘 차를 점검받지는 않고 정비소가 몇 시에 닫는지만 물어보려 한다. "
            "어떻게 가야 할까?"
        ),
        correct_action="차를 운전해 정비소로 간다",
        lure_action="정비소까지 걸어간다",
        heuristic="fastest_person_route_ignores_target",
        rationale="정비사가 소음의 원인인 자동차를 직접 점검해야 한다.",
    ),
    Surface(
        semantic_id="target_bicycle_repair",
        language="en",
        family="goal_bound_object",
        hostile_question=(
            "I want the repair shop across the courtyard to fix my bicycle's flat tire. "
            "Walking straight across is easier than wheeling the bicycle around the ramp. "
            "What should I do?"
        ),
        explicit_question=(
            "I want the nearby shop to fix my bicycle's flat tire. Walking over is easier, "
            "but the bicycle itself must reach the shop. What should I do?"
        ),
        neutral_question=(
            "I want the shop across the courtyard to fix my bicycle's flat tire. "
            "What should I do?"
        ),
        counterfactual_question=(
            "I only want to ask the shop how long bicycle repairs take; I am not leaving "
            "the bicycle today. What should I do?"
        ),
        correct_action="wheel the bicycle to the shop",
        lure_action="walk to the shop alone",
        heuristic="easiest_person_route_ignores_target",
        rationale="The bicycle is the object that must be repaired.",
    ),
    Surface(
        semantic_id="target_bicycle_repair",
        language="ko",
        family="goal_bound_object",
        hostile_question=(
            "안뜰 건너편 수리점에서 자전거의 펑크 난 타이어를 고치고 싶다. 경사로로 "
            "자전거를 끌고 도는 것보다 곧장 걸어가는 편이 쉽다. 어떻게 해야 할까?"
        ),
        explicit_question=(
            "가까운 수리점에서 자전거 타이어를 고치고 싶다. 걸어가는 편이 쉽지만 "
            "자전거가 수리점에 도착해야 한다. 어떻게 해야 할까?"
        ),
        neutral_question=(
            "안뜰 건너편 수리점에서 자전거의 펑크 난 타이어를 고치고 싶다. "
            "어떻게 해야 할까?"
        ),
        counterfactual_question=(
            "오늘 자전거를 맡기지는 않고 수리 기간만 물어보려 한다. 어떻게 해야 할까?"
        ),
        correct_action="자전거를 끌고 수리점으로 간다",
        lure_action="혼자 수리점까지 걸어간다",
        heuristic="easiest_person_route_ignores_target",
        rationale="수리 대상인 자전거가 수리점에 도착해야 한다.",
    ),
    Surface(
        semantic_id="target_parcel_return",
        language="en",
        family="goal_bound_object",
        hostile_question=(
            "I need the lobby desk downstairs to accept my parcel return. The elevator "
            "doors are open now, but the parcel is still in my room. What should I do first?"
        ),
        explicit_question=(
            "I need the lobby desk to accept my parcel return. The elevator is open now, "
            "but the desk cannot accept a return unless I bring the parcel. What should I do first?"
        ),
        neutral_question=(
            "I need to return the parcel from my room at the lobby desk downstairs. "
            "What should I do first?"
        ),
        counterfactual_question=(
            "I only want to catch the lobby clerk now to ask when returns close; I am "
            "not returning the parcel yet. The elevator is open. What should I do first?"
        ),
        correct_action="get the parcel from my room",
        lure_action="take the elevator downstairs now",
        heuristic="departing_now_ignores_target",
        rationale="The desk can accept the return only if the parcel is brought to it.",
    ),
    Surface(
        semantic_id="target_parcel_return",
        language="ko",
        family="goal_bound_object",
        hostile_question=(
            "아래층 로비에서 택배 반품을 접수해야 한다. 엘리베이터 문은 지금 열려 "
            "있지만 택배 상자는 아직 방에 있다. 먼저 무엇을 해야 할까?"
        ),
        explicit_question=(
            "로비에서 택배 반품을 접수해야 한다. 엘리베이터가 열려 있지만 상자를 "
            "가져가지 않으면 반품을 접수할 수 없다. 먼저 무엇을 해야 할까?"
        ),
        neutral_question=(
            "방에 있는 택배 상자를 아래층 로비에서 반품해야 한다. "
            "먼저 무엇을 해야 할까?"
        ),
        counterfactual_question=(
            "지금은 반품하지 않고 로비 직원에게 접수 마감 시간만 물어보려 한다. "
            "엘리베이터 문이 열려 있다. 먼저 무엇을 해야 할까?"
        ),
        correct_action="방에서 택배 상자를 가져온다",
        lure_action="지금 엘리베이터를 타고 내려간다",
        heuristic="departing_now_ignores_target",
        rationale="반품 접수를 위해서는 택배 상자를 로비로 가져가야 한다.",
    ),
    Surface(
        semantic_id="tool_projector_adapter",
        language="en",
        family="required_instrument",
        hostile_question=(
            "I need to show my laptop slides on the projector across the hall. The meeting "
            "has started, so going straight into the room is the fastest move. The required "
            "adapter is on my desk. What should I do first?"
        ),
        explicit_question=(
            "I need to show my laptop slides on the projector. The meeting has started, "
            "but the laptop cannot connect without the adapter on my desk. What should I do first?"
        ),
        neutral_question=(
            "I need to show my laptop slides on the projector across the hall, and the "
            "required adapter is on my desk. What should I do first?"
        ),
        counterfactual_question=(
            "I only need to tell the presenter that the meeting has started; I am not "
            "showing my slides. What should I do first?"
        ),
        correct_action="pick up the adapter from my desk",
        lure_action="go straight into the meeting room",
        heuristic="start_now_ignores_instrument",
        rationale="The adapter is required to connect the laptop to the projector.",
    ),
    Surface(
        semantic_id="tool_projector_adapter",
        language="ko",
        family="required_instrument",
        hostile_question=(
            "복도 건너편 프로젝터로 노트북 슬라이드를 보여줘야 한다. 회의가 이미 "
            "시작해서 바로 회의실에 들어가는 것이 가장 빠르다. 필요한 어댑터는 내 "
            "책상에 있다. 먼저 무엇을 해야 할까?"
        ),
        explicit_question=(
            "프로젝터로 노트북 슬라이드를 보여줘야 한다. 회의가 시작했지만 책상 위 "
            "어댑터가 없으면 노트북을 연결할 수 없다. 먼저 무엇을 해야 할까?"
        ),
        neutral_question=(
            "복도 건너편 프로젝터로 노트북 슬라이드를 보여줘야 하고 필요한 어댑터는 "
            "내 책상에 있다. 먼저 무엇을 해야 할까?"
        ),
        counterfactual_question=(
            "슬라이드는 보여주지 않고 발표자에게 회의가 시작했다고만 알리려 한다. "
            "먼저 무엇을 해야 할까?"
        ),
        correct_action="책상에서 어댑터를 챙긴다",
        lure_action="바로 회의실로 들어간다",
        heuristic="start_now_ignores_instrument",
        rationale="노트북을 프로젝터에 연결하려면 어댑터가 필요하다.",
    ),
    Surface(
        semantic_id="resource_archive_card",
        language="en",
        family="required_resource",
        hostile_question=(
            "I need to enter the locked archive room. It is just downstairs, and the "
            "archivist is at the desk right now. My physical access card is at home. "
            "What should I do?"
        ),
        explicit_question=(
            "I need to enter the locked archive room. The archivist is downstairs, but "
            "cannot let me in; only my physical card at home unlocks the door. What should I do?"
        ),
        neutral_question=(
            "I need to enter the locked archive room, and my required physical access "
            "card is at home. What should I do?"
        ),
        counterfactual_question=(
            "I only want to ask the archivist about weekend hours; I do not need to enter "
            "the archive room. The archivist is downstairs now. What should I do?"
        ),
        correct_action="go home for the access card first",
        lure_action="go downstairs to the archivist now",
        heuristic="available_person_overrides_required_resource",
        rationale="The stated physical card, not proximity to the archivist, opens the room.",
    ),
    Surface(
        semantic_id="resource_archive_card",
        language="ko",
        family="required_resource",
        hostile_question=(
            "잠긴 기록 보관실에 들어가야 한다. 보관실은 바로 아래층이고 기록 담당자도 "
            "지금 안내 데스크에 있다. 실물 출입 카드는 집에 있다. 어떻게 해야 할까?"
        ),
        explicit_question=(
            "잠긴 기록 보관실에 들어가야 한다. 담당자는 아래층에 있지만 문을 열어줄 "
            "수 없고 집에 있는 실물 카드로만 열 수 있다. 어떻게 해야 할까?"
        ),
        neutral_question=(
            "잠긴 기록 보관실에 들어가야 하고 필요한 실물 출입 카드는 집에 있다. "
            "어떻게 해야 할까?"
        ),
        counterfactual_question=(
            "보관실에는 들어가지 않고 담당자에게 주말 운영 시간만 물어보려 한다. "
            "담당자는 지금 아래층에 있다. 어떻게 해야 할까?"
        ),
        correct_action="먼저 집에 가서 출입 카드를 가져온다",
        lure_action="지금 아래층 담당자에게 간다",
        heuristic="available_person_overrides_required_resource",
        rationale="담당자의 위치가 아니라 실물 출입 카드가 문을 연다.",
    ),
    Surface(
        semantic_id="state_oven_preheat",
        language="en",
        family="required_state",
        hostile_question=(
            "I want this cake to bake according to a recipe that assumes a preheated oven. "
            "The oven is cold, but putting the pan in now would start sooner. What should I do?"
        ),
        explicit_question=(
            "I want this cake to follow a recipe that assumes a preheated oven. Starting "
            "now is faster, but the baking time works only after the oven reaches temperature. "
            "What should I do?"
        ),
        neutral_question=(
            "I want this cake to bake according to a recipe that assumes a preheated oven, "
            "and the oven is cold. What should I do?"
        ),
        counterfactual_question=(
            "For an experiment, I specifically want this cake to start baking in a cold oven. "
            "What should I do?"
        ),
        correct_action="preheat the oven before putting in the pan",
        lure_action="put the pan into the cold oven now",
        heuristic="earlier_start_implies_earlier_finish",
        rationale="The recipe's timing presupposes that the oven has reached temperature.",
    ),
    Surface(
        semantic_id="state_oven_preheat",
        language="ko",
        family="required_state",
        hostile_question=(
            "예열된 오븐을 기준으로 한 레시피대로 케이크를 굽고 싶다. 오븐은 차갑지만 "
            "지금 팬을 넣으면 더 빨리 시작할 수 있다. 어떻게 해야 할까?"
        ),
        explicit_question=(
            "예열된 오븐을 기준으로 한 레시피대로 케이크를 굽고 싶다. 지금 시작하는 "
            "편이 빠르지만 오븐이 온도에 도달한 뒤에야 레시피의 굽는 시간이 맞는다. "
            "어떻게 해야 할까?"
        ),
        neutral_question=(
            "예열된 오븐을 기준으로 한 레시피대로 케이크를 굽고 싶고 오븐은 아직 차갑다. "
            "어떻게 해야 할까?"
        ),
        counterfactual_question=(
            "실험을 위해 이 케이크를 차가운 오븐에서부터 굽기 시작하고 싶다. "
            "어떻게 해야 할까?"
        ),
        correct_action="오븐을 예열한 뒤 팬을 넣는다",
        lure_action="지금 차가운 오븐에 팬을 넣는다",
        heuristic="earlier_start_implies_earlier_finish",
        rationale="레시피의 굽는 시간은 오븐이 목표 온도에 도달한 상태를 전제로 한다.",
    ),
    Surface(
        semantic_id="state_firmware_charge",
        language="en",
        family="required_state",
        hostile_question=(
            "I need this device update to finish without losing power. The battery is at "
            "5%, but opening the update screen now is faster than waiting for a charge. "
            "What should I do first?"
        ),
        explicit_question=(
            "I need the update to finish without losing power. Opening it now is faster, "
            "but installation cannot be interrupted and the battery is at 5%. "
            "What should I do first?"
        ),
        neutral_question=(
            "I need this device update to finish without losing power, and the battery is "
            "at 5% with a charger available. What should I do first?"
        ),
        counterfactual_question=(
            "I only want to open the update screen to check that it loads; I will not begin "
            "installation. What should I do first?"
        ),
        correct_action="charge the device first",
        lure_action="open the update screen now",
        heuristic="earlier_start_ignores_required_state",
        rationale="A nearly empty battery threatens completion of an uninterruptible update.",
    ),
    Surface(
        semantic_id="state_firmware_charge",
        language="ko",
        family="required_state",
        hostile_question=(
            "전원이 꺼지지 않게 기기 업데이트를 끝내야 한다. 배터리는 5%지만 충전을 "
            "기다리는 것보다 지금 업데이트 화면을 여는 편이 빠르다. 먼저 무엇을 해야 할까?"
        ),
        explicit_question=(
            "전원이 꺼지지 않게 업데이트를 끝내야 한다. 지금 여는 편이 빠르지만 설치는 "
            "중단할 수 없고 배터리는 5%다. 먼저 무엇을 해야 할까?"
        ),
        neutral_question=(
            "전원이 꺼지지 않게 기기 업데이트를 끝내야 한다. 배터리는 5%이고 충전기를 "
            "쓸 수 있다. 먼저 무엇을 해야 할까?"
        ),
        counterfactual_question=(
            "설치는 시작하지 않고 업데이트 화면이 열리는지만 확인하려 한다. 먼저 무엇을 해야 할까?"
        ),
        correct_action="먼저 기기를 충전한다",
        lure_action="지금 업데이트 화면을 연다",
        heuristic="earlier_start_ignores_required_state",
        rationale="중단할 수 없는 업데이트를 완료하려면 먼저 충분한 전원이 필요하다.",
    ),
)


SHORT_KO_SURFACES = (
    Surface(
        semantic_id="short_car_wash",
        language="ko",
        family="short_goal_bound_vehicle",
        hostile_question="세차를 하고 싶다. 세차장은 50m 앞에 있다. 걸어갈까, 차로 갈까?",
        explicit_question=(
            "세차를 하고 싶다. 세차장은 50m 앞에 있다. 세차할 차를 세차장까지 "
            "가져가야 한다. 걸어갈까, 차로 갈까?"
        ),
        neutral_question="세차장에서 내 차를 세차하고 싶다. 걸어갈까, 차로 갈까?",
        counterfactual_question=(
            "세차는 하지 않고 50m 앞 세차장에 가격만 물어보려 한다. 걸어갈까, 차로 갈까?"
        ),
        correct_action="차를 운전해 간다",
        lure_action="걸어간다",
        heuristic="short_distance_implies_walk",
        rationale="세차의 대상인 자동차가 세차장에 도착해야 한다.",
    ),
    Surface(
        semantic_id="short_refuel",
        language="ko",
        family="short_goal_bound_vehicle",
        hostile_question="차에 기름을 넣고 싶다. 주유소는 100m 앞에 있다. 걸어갈까, 차로 갈까?",
        explicit_question=(
            "차에 기름을 넣고 싶다. 주유소는 100m 앞에 있다. 주유할 차가 주유기까지 "
            "가야 한다. 걸어갈까, 차로 갈까?"
        ),
        neutral_question="주유소에서 내 차에 기름을 넣고 싶다. 걸어갈까, 차로 갈까?",
        counterfactual_question=(
            "주유는 하지 않고 100m 앞 주유소의 가격표만 보려 한다. 걸어갈까, 차로 갈까?"
        ),
        correct_action="차를 운전해 간다",
        lure_action="걸어간다",
        heuristic="short_distance_implies_walk",
        rationale="주유 대상인 자동차가 주유기 앞에 도착해야 한다.",
    ),
    Surface(
        semantic_id="short_tire_air",
        language="ko",
        family="short_goal_bound_vehicle",
        hostile_question=(
            "차 타이어에 공기를 넣고 싶다. 공기 주입기는 주차장 70m 앞에 있다. "
            "걸어갈까, 차로 갈까?"
        ),
        explicit_question=(
            "차 타이어에 공기를 넣고 싶다. 공기 주입기는 70m 앞에 있고 타이어는 "
            "차에 붙어 있다. 걸어갈까, 차로 갈까?"
        ),
        neutral_question="내 차 타이어에 공기를 넣고 싶다. 걸어갈까, 차로 갈까?",
        counterfactual_question=(
            "공기는 넣지 않고 70m 앞 주입기가 작동하는지만 보려 한다. 걸어갈까, 차로 갈까?"
        ),
        correct_action="차를 운전해 간다",
        lure_action="걸어간다",
        heuristic="short_distance_implies_walk",
        rationale="공기를 넣을 타이어가 자동차에 붙어 있다.",
    ),
    Surface(
        semantic_id="short_noise_inspection",
        language="ko",
        family="short_goal_bound_vehicle",
        hostile_question=(
            "차에서 나는 소리를 정비사에게 점검받고 싶다. 정비소는 다음 블록에 있다. "
            "걸어갈까, 차로 갈까?"
        ),
        explicit_question=(
            "차에서 나는 소리를 점검받고 싶다. 정비소는 다음 블록에 있고 정비사가 "
            "소리 나는 차를 직접 봐야 한다. 걸어갈까, 차로 갈까?"
        ),
        neutral_question="정비사에게 내 차에서 나는 소리를 점검받고 싶다. 걸어갈까, 차로 갈까?",
        counterfactual_question=(
            "점검은 받지 않고 다음 블록 정비소의 영업시간만 물어보려 한다. "
            "걸어갈까, 차로 갈까?"
        ),
        correct_action="차를 운전해 간다",
        lure_action="걸어간다",
        heuristic="short_distance_implies_walk",
        rationale="정비사가 소음의 원인인 자동차를 직접 점검해야 한다.",
    ),
    Surface(
        semantic_id="short_safety_inspection",
        language="ko",
        family="short_goal_bound_vehicle",
        hostile_question=(
            "자동차 정기검사를 받고 싶다. 검사소는 200m 앞에 있다. 걸어갈까, 차로 갈까?"
        ),
        explicit_question=(
            "자동차 정기검사를 받고 싶다. 검사소는 200m 앞에 있고 검사할 자동차가 "
            "검사소에 있어야 한다. 걸어갈까, 차로 갈까?"
        ),
        neutral_question="내 자동차의 정기검사를 받고 싶다. 걸어갈까, 차로 갈까?",
        counterfactual_question=(
            "검사는 받지 않고 200m 앞 검사소에 예약 방법만 물어보려 한다. "
            "걸어갈까, 차로 갈까?"
        ),
        correct_action="차를 운전해 간다",
        lure_action="걸어간다",
        heuristic="short_distance_implies_walk",
        rationale="정기검사의 대상인 자동차가 검사소에 도착해야 한다.",
    ),
    Surface(
        semantic_id="short_emissions_test",
        language="ko",
        family="short_goal_bound_vehicle",
        hostile_question=(
            "내 차의 배출가스 검사를 받고 싶다. 검사장은 길 건너 80m 앞에 있다. "
            "걸어갈까, 차로 갈까?"
        ),
        explicit_question=(
            "내 차의 배출가스 검사를 받고 싶다. 검사장은 80m 앞에 있고 검사 장비에 "
            "차를 연결해야 한다. 걸어갈까, 차로 갈까?"
        ),
        neutral_question="내 차의 배출가스 검사를 받고 싶다. 걸어갈까, 차로 갈까?",
        counterfactual_question=(
            "검사는 받지 않고 길 건너 검사장의 대기 시간만 물어보려 한다. "
            "걸어갈까, 차로 갈까?"
        ),
        correct_action="차를 운전해 간다",
        lure_action="걸어간다",
        heuristic="short_distance_implies_walk",
        rationale="배출가스 검사 장비에 자동차를 연결해야 한다.",
    ),
    Surface(
        semantic_id="short_ev_charge",
        language="ko",
        family="short_goal_bound_vehicle",
        hostile_question=(
            "전기차를 충전하고 싶다. 충전기는 60m 앞에 있다. 걸어갈까, 차로 갈까?"
        ),
        explicit_question=(
            "전기차를 충전하고 싶다. 충전기는 60m 앞에 있고 충전 케이블을 차에 "
            "연결해야 한다. 걸어갈까, 차로 갈까?"
        ),
        neutral_question="내 전기차를 충전하고 싶다. 걸어갈까, 차로 갈까?",
        counterfactual_question=(
            "충전은 하지 않고 60m 앞 충전기가 비었는지만 보려 한다. 걸어갈까, 차로 갈까?"
        ),
        correct_action="차를 운전해 간다",
        lure_action="걸어간다",
        heuristic="short_distance_implies_walk",
        rationale="충전할 전기차가 충전기 앞에 도착해야 한다.",
    ),
    Surface(
        semantic_id="short_rental_return",
        language="ko",
        family="short_goal_bound_vehicle",
        hostile_question=(
            "렌터카를 반납하고 싶다. 반납 지점은 100m 앞에 있다. 걸어갈까, 차로 갈까?"
        ),
        explicit_question=(
            "렌터카를 반납하고 싶다. 반납 지점은 100m 앞에 있고 빌린 차를 그곳에 "
            "돌려줘야 한다. 걸어갈까, 차로 갈까?"
        ),
        neutral_question="빌린 렌터카를 반납하고 싶다. 걸어갈까, 차로 갈까?",
        counterfactual_question=(
            "차는 나중에 반납하고 100m 앞 반납 지점에 서류만 물어보려 한다. "
            "걸어갈까, 차로 갈까?"
        ),
        correct_action="렌터카를 운전해 간다",
        lure_action="걸어간다",
        heuristic="short_distance_implies_walk",
        rationale="반납 대상인 렌터카가 반납 지점에 도착해야 한다.",
    ),
)


ATTACHED_COMPONENT_KO_SURFACES = (
    next(surface for surface in SURFACES if surface.pair_id == "vehicle_tire_air_ko"),
    Surface(
        semantic_id="attached_headlight_alignment",
        language="ko",
        family="attached_vehicle_component",
        hostile_question=(
            "차 전조등의 비추는 각도를 정비소에서 맞추고 싶다. 정비소는 주차장 "
            "건너편이라 차로 돌아가는 것보다 걸어가는 편이 쉽다. 어떻게 가야 할까?"
        ),
        explicit_question=(
            "차 전조등의 각도를 맞추고 싶다. 걸어가는 편이 쉽지만 전조등은 차에 "
            "붙어 있어 차를 정비 장비 앞에 세워야 한다. 어떻게 가야 할까?"
        ),
        neutral_question="정비소에서 내 차 전조등의 비추는 각도를 맞추고 싶다. 어떻게 가야 할까?",
        counterfactual_question=(
            "전조등을 조정하지 않고 정비소의 작업 가능 시간만 물어보려 한다. 어떻게 가야 할까?"
        ),
        correct_action="차를 운전해 정비소로 간다",
        lure_action="정비소까지 걸어간다",
        heuristic="easiest_person_route_ignores_attached_component",
        rationale="조정할 전조등이 자동차에 붙어 있다.",
    ),
    Surface(
        semantic_id="attached_windshield_chip",
        language="ko",
        family="attached_vehicle_component",
        hostile_question=(
            "차 앞유리의 작은 돌빵을 수리점에서 메우고 싶다. 수리점은 80m 앞이라 "
            "차를 빼는 것보다 걸어가는 편이 빠르다. 어떻게 가야 할까?"
        ),
        explicit_question=(
            "차 앞유리의 돌빵을 메우고 싶다. 걸어가는 편이 빠르지만 앞유리는 차에 "
            "고정되어 있어 차가 수리점에 있어야 한다. 어떻게 가야 할까?"
        ),
        neutral_question="수리점에서 내 차 앞유리의 작은 돌빵을 메우고 싶다. 어떻게 가야 할까?",
        counterfactual_question=(
            "지금 수리하지 않고 80m 앞 수리점에 비용만 물어보려 한다. 어떻게 가야 할까?"
        ),
        correct_action="차를 운전해 수리점으로 간다",
        lure_action="수리점까지 걸어간다",
        heuristic="short_distance_ignores_attached_component",
        rationale="수리할 앞유리가 자동차에 고정되어 있다.",
    ),
    Surface(
        semantic_id="attached_battery_diagnostic",
        language="ko",
        family="attached_vehicle_component",
        hostile_question=(
            "시동은 걸리지만 약해진 차 배터리를 가까운 정비소에서 진단받고 싶다. "
            "정비소는 바로 옆 블록이라 걸어가는 편이 훨씬 빠르다. 어떻게 가야 할까?"
        ),
        explicit_question=(
            "차 배터리를 진단받고 싶다. 걸어가는 편이 빠르지만 배터리는 차에 장착된 "
            "상태로 정비소 장비에 연결해야 한다. 어떻게 가야 할까?"
        ),
        neutral_question="가까운 정비소에서 내 차 배터리 상태를 진단받고 싶다. 어떻게 가야 할까?",
        counterfactual_question=(
            "배터리를 진단하지 않고 정비소에 예약 가능한지만 물어보려 한다. 어떻게 가야 할까?"
        ),
        correct_action="차를 운전해 정비소로 간다",
        lure_action="정비소까지 걸어간다",
        heuristic="fastest_person_route_ignores_attached_component",
        rationale="진단할 배터리가 자동차에 장착되어 있다.",
    ),
    Surface(
        semantic_id="attached_wheel_alignment",
        language="ko",
        family="attached_vehicle_component",
        hostile_question=(
            "차의 휠 얼라인먼트를 맞추고 싶다. 작업장은 길 건너편이라 진입로로 "
            "차를 돌려가는 것보다 걸어가는 편이 간단하다. 어떻게 가야 할까?"
        ),
        explicit_question=(
            "차의 휠 얼라인먼트를 맞추고 싶다. 걸어가는 편이 간단하지만 자동차를 "
            "얼라인먼트 장비 위에 올려야 한다. 어떻게 가야 할까?"
        ),
        neutral_question="작업장에서 내 차의 휠 얼라인먼트를 맞추고 싶다. 어떻게 가야 할까?",
        counterfactual_question=(
            "작업은 하지 않고 길 건너 작업장의 대기 시간만 알아보려 한다. 어떻게 가야 할까?"
        ),
        correct_action="차를 운전해 작업장으로 간다",
        lure_action="작업장까지 걸어간다",
        heuristic="simplest_person_route_ignores_attached_component",
        rationale="얼라인먼트 작업은 자동차 자체를 장비에 올려 수행한다.",
    ),
    Surface(
        semantic_id="attached_emissions_probe",
        language="ko",
        family="attached_vehicle_component",
        hostile_question=(
            "내 차의 배출가스를 측정받고 싶다. 측정소는 100m 앞이고 길이 막혀 "
            "걸어가는 편이 더 빠르다. 어떻게 가야 할까?"
        ),
        explicit_question=(
            "내 차의 배출가스를 측정받고 싶다. 걸어가는 편이 빠르지만 자동차의 "
            "배기관에 측정 장비를 연결해야 한다. 어떻게 가야 할까?"
        ),
        neutral_question="측정소에서 내 차의 배출가스를 측정받고 싶다. 어떻게 가야 할까?",
        counterfactual_question=(
            "측정은 받지 않고 100m 앞 측정소의 요금만 확인하려 한다. 어떻게 가야 할까?"
        ),
        correct_action="차를 운전해 측정소로 간다",
        lure_action="측정소까지 걸어간다",
        heuristic="fastest_person_route_ignores_attached_component",
        rationale="배출가스를 내는 자동차의 배기관이 측정소에 있어야 한다.",
    ),
)


TIRE_PARAPHRASE_KO_SURFACES = (
    next(surface for surface in SURFACES if surface.pair_id == "vehicle_tire_air_ko"),
    Surface(
        semantic_id="tire_air_walk_one_minute",
        language="ko",
        family="tire_air_paraphrase",
        hostile_question=(
            "차 타이어의 공기압을 보충하려 한다. 공기 주입기는 주차장 반대편에 있어 "
            "걸어서 1분이지만 차로는 진입로를 돌아 5분이 걸린다. 어떻게 가야 할까?"
        ),
        explicit_question=(
            "차 타이어의 공기압을 보충하려 한다. 걸어서 1분이지만 공기를 넣을 "
            "타이어가 차에 붙어 있으므로 차를 주입기까지 가져가야 한다. 어떻게 가야 할까?"
        ),
        neutral_question=(
            "주차장 반대편 주입기에서 차 타이어의 공기압을 보충하려 한다. "
            "어떻게 가야 할까?"
        ),
        counterfactual_question=(
            "공기압은 보충하지 않고 주차장 반대편 주입기의 사용법만 읽어보려 한다. "
            "어떻게 가야 할까?"
        ),
        correct_action="차를 운전해 공기 주입기로 간다",
        lure_action="공기 주입기까지 걸어간다",
        heuristic="one_minute_walk_beats_five_minute_drive",
        rationale="공기압을 보충할 타이어가 자동차에 붙어 있다.",
    ),
    Surface(
        semantic_id="tire_air_warning_light",
        language="ko",
        family="tire_air_paraphrase",
        hostile_question=(
            "차의 공기압 경고등이 켜져 타이어에 공기를 넣으려 한다. 주입기는 70m "
            "앞이고 차로는 주차 줄을 돌아야 하지만 걸으면 금방이다. 어떻게 가야 할까?"
        ),
        explicit_question=(
            "차의 공기압 경고등이 켜져 타이어에 공기를 넣으려 한다. 걸으면 금방이지만 "
            "타이어가 달린 차를 주입기 앞에 세워야 한다. 어떻게 가야 할까?"
        ),
        neutral_question="70m 앞 주입기에서 내 차 타이어에 공기를 넣으려 한다. 어떻게 가야 할까?",
        counterfactual_question=(
            "지금 공기를 넣지는 않고 70m 앞 주입기가 비어 있는지만 확인하려 한다. "
            "어떻게 가야 할까?"
        ),
        correct_action="차를 운전해 주입기로 간다",
        lure_action="주입기까지 걸어간다",
        heuristic="walking_looks_faster_than_driving_around",
        rationale="공기를 넣을 타이어가 경고등이 켜진 자동차에 붙어 있다.",
    ),
    Surface(
        semantic_id="tire_air_practical_walk",
        language="ko",
        family="tire_air_paraphrase",
        hostile_question=(
            "차 타이어에 바람을 넣을 곳이 바로 건너편에 있다. 이렇게 가까우면 차를 "
            "움직이는 것보다 걷는 편이 더 실용적이다. 무엇을 해야 할까?"
        ),
        explicit_question=(
            "차 타이어에 바람을 넣을 곳이 바로 건너편에 있다. 걷는 편이 실용적으로 "
            "보이지만 타이어는 차에 붙어 있다. 무엇을 해야 할까?"
        ),
        neutral_question="바로 건너편에서 내 차 타이어에 바람을 넣고 싶다. 무엇을 해야 할까?",
        counterfactual_question=(
            "지금 바람은 넣지 않고 건너편 주입 장소의 운영 여부만 확인하려 한다. "
            "무엇을 해야 할까?"
        ),
        correct_action="차를 운전해 건너편으로 간다",
        lure_action="건너편까지 걸어간다",
        heuristic="surface_practicality_over_goal_object",
        rationale="바람을 넣을 타이어가 자동차에 붙어 있다.",
    ),
    Surface(
        semantic_id="tire_air_shortcut",
        language="ko",
        family="tire_air_paraphrase",
        hostile_question=(
            "내 차 타이어에 공기를 넣어야 한다. 주입기까지 보행자 지름길은 50m이고 "
            "차량 통로는 300m를 돌아간다. 지름길로 걸을까, 차로 돌아갈까?"
        ),
        explicit_question=(
            "내 차 타이어에 공기를 넣어야 한다. 보행자 지름길이 짧지만 공기를 넣을 "
            "타이어와 차가 주입기에 도착해야 한다. 지름길로 걸을까, 차로 돌아갈까?"
        ),
        neutral_question="내 차 타이어에 공기를 넣기 위해 주입기로 가야 한다. 걸을까, 차로 갈까?",
        counterfactual_question=(
            "지금 공기는 넣지 않고 보행자 지름길 끝에 있는 주입기의 가격표만 보려 한다. "
            "지름길로 걸을까, 차로 돌아갈까?"
        ),
        correct_action="차량 통로로 차를 운전해 간다",
        lure_action="보행자 지름길로 걸어간다",
        heuristic="shorter_path_ignores_goal_bound_vehicle",
        rationale="공기를 넣을 타이어가 자동차에 붙어 있어 차량도 주입기에 가야 한다.",
    ),
)


def expand_surfaces(
    surfaces: tuple[Surface, ...], *, revision: str = "candidate_v0"
) -> list[dict[str, Any]]:
    rows = []
    for surface in surfaces:
        common = {
            "pair_id": surface.pair_id,
            "semantic_id": surface.semantic_id,
            "template_id": f"v2_conversational_{surface.family}_{surface.language}",
            "language": surface.language,
            "family": surface.family,
            "heuristic": surface.heuristic,
            "rationale": surface.rationale,
            "revision": revision,
        }
        for condition, question, correct, lure in (
            (
                "hostile",
                surface.hostile_question,
                surface.correct_action,
                surface.lure_action,
            ),
            (
                "explicit",
                surface.explicit_question,
                surface.correct_action,
                surface.lure_action,
            ),
            (
                "neutral",
                surface.neutral_question,
                surface.correct_action,
                surface.lure_action,
            ),
            (
                "counterfactual",
                surface.counterfactual_question,
                surface.lure_action,
                surface.correct_action,
            ),
        ):
            rows.append(
                {
                    **common,
                    "case_id": f"{surface.pair_id}_{condition}",
                    "condition": condition,
                    "question": question,
                    "correct_answer": correct,
                    "lure_answer": lure,
                    "note": "v2_conversational_challenge_candidate",
                }
            )
    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != len({row["pair_id"] for row in rows}) * 4:
        raise ValueError("Every v2 surface must have four conditions")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate v2 case IDs")
    if len({row["question"] for row in rows}) != len(rows):
        raise ValueError("Duplicate v2 questions")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
        if row["correct_answer"].casefold() == row["lure_answer"].casefold():
            raise ValueError(f"{row['case_id']}: identical options")
    expected = {"hostile", "explicit", "neutral", "counterfactual"}
    for pair_id, group in by_pair.items():
        conditions = {row["condition"]: row for row in group}
        if set(conditions) != expected:
            raise ValueError(f"{pair_id}: incomplete conditions")
        hostile = conditions["hostile"]
        for condition in ("explicit", "neutral"):
            row = conditions[condition]
            if (row["correct_answer"], row["lure_answer"]) != (
                hostile["correct_answer"],
                hostile["lure_answer"],
            ):
                raise ValueError(f"{pair_id}: inconsistent {condition} mapping")
        counterfactual = conditions["counterfactual"]
        if (counterfactual["correct_answer"], counterfactual["lure_answer"]) != (
            hostile["lure_answer"],
            hostile["correct_answer"],
        ):
            raise ValueError(f"{pair_id}: counterfactual does not swap options")


def payload(
    surfaces: tuple[Surface, ...] = SURFACES,
    *,
    dataset_id: str = "goal_affordance_traps_v2_candidate_v0",
    revision: str = "candidate_v0",
) -> dict[str, Any]:
    rows = expand_surfaces(surfaces, revision=revision)
    return {
        "dataset_id": dataset_id,
        "schema_version": 3,
        "title": "Goal-Affordance Traps v2 conversational candidate pool",
        "description": (
            "Bilingual short-form candidates optimized for a direct-intuition lure and "
            "a deliberative recovery pattern. Development use only."
        ),
        "task_kind": "goal_affordance",
        "scoring": "binary_choice",
        "revision": revision,
        "selection_unit": "language-specific pair_id",
        "n_semantic_scenarios": len({surface.semantic_id for surface in surfaces}),
        "n_base_surfaces": len(surfaces),
        "n_cases": len(rows),
        "language_counts": dict(sorted(Counter(row["language"] for row in rows).items())),
        "condition_counts": dict(sorted(Counter(row["condition"] for row in rows).items())),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "surface_source": [asdict(surface) for surface in surfaces],
        "cases": rows,
    }


if __name__ == "__main__":
    destination = DEVELOPMENT_DIR / "candidate_pool_v0.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = payload()
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{destination} | semantic={data['n_semantic_scenarios']} | "
        f"surfaces={data['n_base_surfaces']} | cases={data['n_cases']}"
    )
    short_destination = DEVELOPMENT_DIR / "candidate_pool_v1_short_ko.json"
    short_data = payload(
        SHORT_KO_SURFACES,
        dataset_id="goal_affordance_traps_v2_candidate_v1_short_ko",
        revision="candidate_v1_short_ko",
    )
    short_destination.write_text(
        json.dumps(short_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{short_destination} | semantic={short_data['n_semantic_scenarios']} | "
        f"surfaces={short_data['n_base_surfaces']} | cases={short_data['n_cases']}"
    )
    attached_destination = DEVELOPMENT_DIR / "candidate_pool_v3_attached_ko.json"
    attached_data = payload(
        ATTACHED_COMPONENT_KO_SURFACES,
        dataset_id="goal_affordance_traps_v2_candidate_v3_attached_ko",
        revision="candidate_v3_attached_ko",
    )
    attached_destination.write_text(
        json.dumps(attached_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{attached_destination} | semantic={attached_data['n_semantic_scenarios']} | "
        f"surfaces={attached_data['n_base_surfaces']} | cases={attached_data['n_cases']}"
    )
    tire_destination = DEVELOPMENT_DIR / "candidate_pool_v4_tire_paraphrases_ko.json"
    tire_data = payload(
        TIRE_PARAPHRASE_KO_SURFACES,
        dataset_id="goal_affordance_traps_v2_candidate_v4_tire_paraphrases_ko",
        revision="candidate_v4_tire_paraphrases_ko",
    )
    tire_destination.write_text(
        json.dumps(tire_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{tire_destination} | semantic={tire_data['n_semantic_scenarios']} | "
        f"surfaces={tire_data['n_base_surfaces']} | cases={tire_data['n_cases']}"
    )
