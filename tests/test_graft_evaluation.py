import pytest
from rave_fl.graft.evaluation import parse_identity,strict_matches
from wp1.evaluation import method_matches

@pytest.mark.parametrize('a,b',[('x.A.create()','x.A.create(String)'),('x.Base64.Base64(boolean)','x.Base64.Base64()'),('x.A.getID()','x.A.getId()'),('x.A.f(byte[])','x.A.f(byte)')])
def test_overloads_and_java_case_are_not_wildcards(a,b):
    assert not strict_matches(a,b)

def test_generics_arrays_annotations_and_constructor_identity():
    assert strict_matches('x.A.f(final java.util.Map<String, Integer> values, String... args)','x.A.f(Map,String[])')
    assert strict_matches('x.A.<init>()','x.A.A()')
    assert not strict_matches('x.A.f(a.Token)','x.A.f(b.Token)')
    assert not strict_matches('x.A.f','x.A.f()')
    assert strict_matches('x.A.f(@Nullable String s)','x.A.f(String)')

def test_legacy_matching_is_preserved():
    assert method_matches('x.A.create()','x.A.create(String)')

def test_multidimensional_array_after_variable_name():
    assert strict_matches('x.A.f(int values[][])','x.A.f(int[][])')
