"""Coverage contexts for test bodies, excluding native test-runner wrappers."""
from pathlib import PurePath
from coverage import CoveragePlugin


class TestContexts(CoveragePlugin):
    def dynamic_context(self, frame):
        code = frame.f_code
        if not code.co_name.startswith('test'):
            return None
        path = PurePath(code.co_filename)
        if not (path.name.startswith('test_') or path.name.endswith('_test.py')):
            return None
        owner = type(frame.f_locals['self']).__name__ + '.' if 'self' in frame.f_locals else ''
        return str(path) + '::' + owner + code.co_name


def coverage_init(reg, options):
    reg.add_dynamic_context(TestContexts())
