from __future__ import annotations

from dataclasses import dataclass
import sys


SPEED = 0.5
SHIFT_START = 480.0
DEADLINE_WINDOW = 30.0
REWARD_PER_ORDER = 10


@dataclass
class Order:
    id: int
    t: float
    sx: float
    sy: float
    ex: float
    ey: float
    is_preorder: bool


@dataclass
class CourierState:
    courier_id: int
    x: float
    y: float
    available_time: float


def manhattan(x1: float, y1: float, x2: float, y2: float) -> float:
    return abs(x1 - x2) + abs(y1 - y2)


def predict_delivery(order: Order, courier: CourierState) -> float:
    to_pickup = manhattan(courier.x, courier.y, order.sx, order.sy) / SPEED
    pickup_to_dropoff = manhattan(order.sx, order.sy, order.ex, order.ey) / SPEED

    if order.is_preorder:
        arrive_at_pickup = courier.available_time + to_pickup
        return max(arrive_at_pickup, order.t) + pickup_to_dropoff

    depart_time = max(courier.available_time, order.t)
    return depart_time + to_pickup + pickup_to_dropoff


def select_courier(order: Order, couriers: list[CourierState]) -> tuple[int, float, int]:
    deadline = order.t + DEADLINE_WINDOW
    best_choice: tuple[float, float, int] | None = None

    for courier in couriers:
        delivery_time = predict_delivery(order, courier)
        if delivery_time > deadline:
            continue

        candidate = (delivery_time, courier.available_time, courier.courier_id)
        if best_choice is None or candidate < best_choice:
            best_choice = candidate

    if best_choice is None:
        return 0, -1.0, 0

    return best_choice[2], best_choice[0], 1


def parse_orders(tokens: list[str]) -> tuple[int, list[Order]]:
    if len(tokens) < 4:
        return 0, []

    n = int(tokens[2])
    m = int(tokens[3])
    order_tokens = tokens[4:]
    order_count = len(order_tokens) // 6
    orders: list[Order] = []

    for idx in range(order_count):
        base = idx * 6
        orders.append(
            Order(
                id=int(order_tokens[base]),
                t=float(order_tokens[base + 1]),
                sx=float(order_tokens[base + 2]),
                sy=float(order_tokens[base + 3]),
                ex=float(order_tokens[base + 4]),
                ey=float(order_tokens[base + 5]),
                is_preorder=idx < m,
            )
        )

    return n, orders


def solve() -> None:
    tokens = sys.stdin.buffer.read().split()
    if not tokens:
        return

    decoded = [token.decode() for token in tokens]
    n, orders = parse_orders(decoded)
    couriers = [
        CourierState(courier_id=i + 1, x=0.0, y=0.0, available_time=SHIFT_START)
        for i in range(n)
    ]

    completed = 0
    output_lines: list[str] = []

    for order in orders:
        courier_id, delivery_time, success = select_courier(order, couriers)

        if success == 1:
            courier = couriers[courier_id - 1]
            courier.x = order.ex
            courier.y = order.ey
            courier.available_time = delivery_time
            completed += 1
            output_lines.append(
                f"{order.id} {courier_id} {delivery_time:.2f} {success}"
            )
        else:
            output_lines.append(f"{order.id} 0 -1.00 0")

    revenue = completed * REWARD_PER_ORDER
    output_lines.append(f"{completed} {revenue:.2f}")
    sys.stdout.write("\n".join(output_lines))


if __name__ == "__main__":
    solve()
