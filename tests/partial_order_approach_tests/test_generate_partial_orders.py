import sys
from pathlib import Path
import pytest
from math import comb

ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(ROOT))

from partial_order_approach import generate_partial_orders, make_blocks_and_fronts

def get_expected_size_of_partial_orders(m: int, p: int) -> int:
    """ Returns the expected number of ideals for given m and p values. """
    return comb(m, m // 2) ** p

@pytest.mark.parametrize("m, p", [(8, 1), (4, 2), (3, 2)])
def test_partial_orders_size(m: int, p: int):
    """Test that the number of generated partial orders matches the theoretical count."""
    V = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"]

    blocks, front_choices_per_block, _ = make_blocks_and_fronts(V, m, p)

    count = sum(1 for _ in generate_partial_orders(blocks, front_choices_per_block))
    expected_size = get_expected_size_of_partial_orders(m, p)

    assert count == expected_size, f"Expected {expected_size} partial orders, got {count} for m={m}, p={p}"