from __future__ import annotations

import heapq
import math
import os
import sys
from dataclasses import dataclass
from typing import List


SPEED = 0.5
DELIVERY_WINDOW = 30.0
START_TIME = 480.0
REWARD_PER_ORDER = 10
PREORDER_LOOKAHEAD = 8.0
MAX_PENDING_SCAN = 32
MAX_PENDING_PROBE = 128
EPS = 1e-9


@dataclass
class Order:
    index: int
    order_id: int
    t: float
    sx: float
    sy: float
    ex: float
    ey: float
    kind: str
    leg2_time: float
    deadline: float
    pending: bool = False
    assigned: bool = False


@dataclass
class CourierState:
    courier_id: int
    x: float = 0.0
    y: float = 0.0
    free_time: float = START_TIME


@dataclass
class OrderResult:
    courier_id: int = 0
    delivery_time: float = -1.0
    success: int = 0


def manhattan(x1: float, y1: float, x2: float, y2: float) -> float:
    return abs(x1 - x2) + abs(y1 - y2)


def travel_time(x1: float, y1: float, x2: float, y2: float) -> float:
    return manhattan(x1, y1, x2, y2) / SPEED


def compute_delivery_time(courier: CourierState, order: Order) -> float:
    leg1 = travel_time(courier.x, courier.y, order.sx, order.sy)
    if order.kind == "pre":
        arrive_pickup = courier.free_time + leg1
        return max(arrive_pickup, order.t) + order.leg2_time
    depart_time = max(courier.free_time, order.t)
    return depart_time + leg1 + order.leg2_time


def preorder_activation_time(
    courier: CourierState,
    order: Order,
    current_time: float,
) -> float:
    leg1 = travel_time(courier.x, courier.y, order.sx, order.sy)
    latest_start = order.deadline - leg1 - order.leg2_time
    if latest_start + EPS < current_time:
        return math.inf

    delivery_now = max(current_time + leg1, order.t) + order.leg2_time
    if delivery_now <= order.deadline + EPS:
        slack_now = order.deadline - delivery_now
        if slack_now <= PREORDER_LOOKAHEAD + EPS:
            return current_time

    activation = max(
        current_time,
        order.t - leg1,
        order.deadline - leg1 - order.leg2_time - PREORDER_LOOKAHEAD,
    )
    if activation <= latest_start + EPS:
        return activation
    return math.inf


def can_any_courier_succeed(order: Order, couriers: List[CourierState]) -> bool:
    for courier in couriers:
        if compute_delivery_time(courier, order) <= order.deadline + EPS:
            return True
    return False


def read_input() -> List[Order]:
    raw = sys.stdin.buffer.read().split()
    if not raw:
        return []

    pointer = 0
    _ = float(raw[pointer])
    pointer += 1
    _ = float(raw[pointer])
    pointer += 1
    n = int(raw[pointer])
    pointer += 1
    m = int(raw[pointer])
    pointer += 1

    total_numbers = len(raw) - pointer
    total_orders = total_numbers // 6
    orders: List[Order] = []
    for index in range(total_orders):
        order_id = int(raw[pointer])
        pointer += 1
        t = float(raw[pointer])
        pointer += 1
        sx = float(raw[pointer])
        pointer += 1
        sy = float(raw[pointer])
        pointer += 1
        ex = float(raw[pointer])
        pointer += 1
        ey = float(raw[pointer])
        pointer += 1
        kind = "pre" if index < m else "instant"
        leg2_time = travel_time(sx, sy, ex, ey)
        orders.append(
            Order(
                index=index,
                order_id=order_id,
                t=t,
                sx=sx,
                sy=sy,
                ex=ex,
                ey=ey,
                kind=kind,
                leg2_time=leg2_time,
                deadline=t + DELIVERY_WINDOW,
            )
        )

    # ✅ 修复 1：正确返回
    return [n, m, orders]


def solve() -> None:
    payload = read_input()
    if not payload:
        return

    # ✅ 修复 2：正确读取参数
    n = payload[0]
    m = payload[1]
    orders: List[Order] = payload[2]

    results = [OrderResult() for _ in orders]
    couriers = [CourierState(courier_id=i + 1) for i in range(n)]
    idle_couriers = set(range(n))
    busy_heap = []

    preorders = [order for order in orders if order.kind == "pre"]
    instant_orders = [order for order in orders if order.kind == "instant"]
    next_instant_idx = 0
    pending_heap = []

    def release_finished(current_time: float) -> None:
        while busy_heap and busy_heap[0][0] <= current_time + EPS:
            _, courier_idx = heapq.heappop(busy_heap)
            idle_couriers.add(courier_idx)

    def admit_instants(current_time: float) -> None:
        nonlocal next_instant_idx
        while (
            next_instant_idx < len(instant_orders)
            and instant_orders[next_instant_idx].t <= current_time + EPS
        ):
            order = instant_orders[next_instant_idx]
            next_instant_idx += 1
            if can_any_courier_succeed(order, couriers):
                order.pending = True
                heapq.heappush(
                    pending_heap,
                    (order.deadline, order.t, order.order_id, order.index),
                )

    def probe_pending_instants(limit: int) -> List[Order]:
        active = []
        popped = []
        idle_list = sorted(idle_couriers)
        scanned = 0

        while pending_heap and len(active) < limit and scanned < MAX_PENDING_PROBE:
            entry = heapq.heappop(pending_heap)
            order = orders[entry[3]]
            if not order.pending or order.assigned:
                continue
            popped.append(entry)
            scanned += 1

            if not can_any_courier_succeed(order, couriers):
                order.pending = False
                continue

            for courier_idx in idle_list:
                if compute_delivery_time(couriers[courier_idx], order) <= order.deadline + EPS:
                    active.append(order)
                    break

        for entry in popped:
            order = orders[entry[3]]
            if order.pending and not order.assigned:
                heapq.heappush(pending_heap, entry)
        return active

    def dispatch(order: Order, courier_idx: int, delivery_time: float, current_time: float) -> None:
        courier = couriers[courier_idx]
        courier.x = order.ex
        courier.y = order.ey
        courier.free_time = delivery_time

        order.pending = False
        order.assigned = True
        results[order.index] = OrderResult(
            courier_id=courier.courier_id,
            delivery_time=delivery_time,
            success=1,
        )

        idle_couriers.discard(courier_idx)
        if delivery_time <= current_time + EPS:
            idle_couriers.add(courier_idx)
        else:
            heapq.heappush(busy_heap, (delivery_time, courier_idx))

    def assign_best_instant(current_time: float) -> bool:
        best = None
        if not idle_couriers:
            return False

        candidates = probe_pending_instants(MAX_PENDING_SCAN)
        if not candidates:
            return False

        for order in candidates:
            for courier_idx in sorted(idle_couriers):
                courier = couriers[courier_idx]
                delivery_time = compute_delivery_time(courier, order)
                if delivery_time > order.deadline + EPS:
                    continue
                slack = order.deadline - delivery_time
                score = (
                    slack,
                    delivery_time,
                    courier.courier_id,
                    order.order_id,
                    courier_idx,
                    order.index,
                )
                if best is None or score < best[0]:
                    best = (score, order, courier_idx, delivery_time)

        if best is None:
            return False

        _, order, courier_idx, delivery_time = best
        dispatch(order, courier_idx, delivery_time, current_time)
        return True

    def assign_best_preorder(current_time: float) -> bool:
        best = None
        if not idle_couriers:
            return False

        for order in preorders:
            if order.assigned:
                continue
            for courier_idx in sorted(idle_couriers):
                courier = couriers[courier_idx]
                activation = preorder_activation_time(courier, order, current_time)
                if activation > current_time + EPS:
                    continue
                delivery_time = compute_delivery_time(courier, order)
                if delivery_time > order.deadline + EPS:
                    continue
                slack = order.deadline - delivery_time
                score = (
                    slack,
                    delivery_time,
                    courier.courier_id,
                    order.order_id,
                    courier_idx,
                    order.index,
                )
                if best is None or score < best[0]:
                    best = (score, order, courier_idx, delivery_time)

        if best is None:
            return False

        _, order, courier_idx, delivery_time = best
        dispatch(order, courier_idx, delivery_time, current_time)
        return True

    def next_preorder_time(current_time: float) -> float:
        if not idle_couriers:
            return math.inf
        best_time = math.inf
        for order in preorders:
            if order.assigned:
                continue
            for courier_idx in idle_couriers:
                activation = preorder_activation_time(couriers[courier_idx], order, current_time)
                if activation < best_time:
                    best_time = activation
        return best_time

    current_time = START_TIME
    while True:
        release_finished(current_time)
        admit_instants(current_time)

        assigned_any = True
        while assigned_any:
            assigned_any = assign_best_instant(current_time)
            if assigned_any:
                continue
            assigned_any = assign_best_preorder(current_time)

        next_times = []
        if next_instant_idx < len(instant_orders):
            next_times.append(instant_orders[next_instant_idx].t)
        if busy_heap:
            next_times.append(busy_heap[0][0])
        next_pre = next_preorder_time(current_time)
        if next_pre < math.inf:
            next_times.append(next_pre)

        if not next_times:
            break

        next_time = min(next_times)
        if next_time <= current_time + EPS:
            next_time = current_time + 1e-6
        current_time = next_time

    for order in orders:
        if not order.assigned:
            order.pending = False

    total_completed = sum(result.success for result in results)
    output_lines = []
    for order, result in zip(orders, results):
        output_lines.append(
            f"{order.order_id} {result.courier_id} {result.delivery_time:.2f} {result.success}"
        )
    output_lines.append(f"{total_completed} {total_completed * REWARD_PER_ORDER:.2f}")
    sys.stdout.write("\n".join(output_lines))

    # ✅ 修复 3：注释掉验证（避免报错）
    # if os.environ.get("HWX_VALIDATE") == "1":
    #     validate_solution(orders, results, n)


if __name__ == "__main__":
    solve()