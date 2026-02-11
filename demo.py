from sanna import sanna_observe, SannaHaltError

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
