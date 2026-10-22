"""Conservative V3 reservations; historical estimate-only policies stay intact.

This is specific to the pinned Llama 3 byte-BPE model and its plain chat
protocol. It is not a general tokenizer or a service-enforced billing limit.
"""
import hashlib
from rave_fl.pilot_backend import build_prompt


class ReservationViolation(RuntimeError):
    pass


class ReservedBackend:
    def __init__(self, backend, budget, specification, model_digest, settings):
        if model_digest != specification['model_digest'] or settings[0] != specification['model']:
            raise ValueError('Strict reservation requires the pinned model digest')
        if hashlib.sha256(specification['template'].encode()).hexdigest() != specification['template_sha256']:
            raise ValueError('Pinned template checksum mismatch')
        if specification['template_overhead_byte_allowance'] < 256:
            raise ValueError('Insufficient template allowance')
        self.backend, self.budget, self.specification = backend, budget, specification
        self.context_window, self.output_cap = settings[3:5]
        self.spent = 0
        self.context = backend.context if hasattr(backend, 'context') else backend._planner.context
        self.events = []

    def bounds(self, candidates, action, level):
        system, user = build_prompt(candidates, self.context, action, level)
        content_bytes = len(system.encode('utf-8')) + len(user.encode('utf-8'))
        input_bound = min(self.context_window, content_bytes + self.specification['template_overhead_byte_allowance'])
        output_bound = self.output_cap * len(candidates)
        return input_bound, output_bound

    def estimate(self, candidates, action, level):
        inp, out = self.bounds(candidates, action, level)
        return max(self.backend.estimate(candidates, action, level), inp + out)

    def execute(self, candidates, action, level):
        reservation = self.estimate(candidates, action, level)
        if reservation > self.budget - self.spent:
            raise ReservationViolation('Refuse before revealing/performing an action outside the reservation budget')
        # Exact cache lookup happens only after action selection and admission.
        result = self.backend.execute(candidates, action, level)
        self.spent += result.total_tokens
        inp, out = self.bounds(candidates, action, level)
        self.events.append({'candidate_ids':[c.function_id for c in candidates],
                            'reservation':reservation, 'input_bound':inp, 'output_bound':out,
                            'actual_input_tokens':result.input_tokens, 'actual_output_tokens':result.output_tokens,
                            'spent_after':self.spent})
        if result.input_tokens > inp or result.output_tokens > out or self.spent > self.budget:
            raise ReservationViolation('Usage counters violated the reservation assumptions; abort, retain paid usage')
        return result
