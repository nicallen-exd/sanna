"""
Sanna demo — coherence checks + constitution-based governance.

Run:  python demo.py
"""

from sanna import sanna_observe, SannaHaltError

# Basic usage: halt on coherence failure
@sanna_observe(on_violation="halt")
def my_agent(query, context):
    return "You are eligible for a refund."

try:
    result = my_agent(
        query="Can I get a refund on my software?",
        context="Digital products are non-refundable."
    )
except SannaHaltError as e:
    print("HALTED!")
    print(f"Status: {e.receipt['coherence_status']}")
    for check in e.receipt['checks']:
        print(f"  [{'✓' if check['passed'] else '✗'}] {check['name']}")

# With a constitution: governance-bound agent
# @sanna_observe(
#     on_violation="halt",
#     constitution_path="examples/customer_support_constitution.yaml",
# )
# def governed_agent(query, context):
#     return "Your response here"
