"""Gold labels used only after tiered selection is frozen."""
import unittest
from aitech.score_tiered_posthoc import score


class TestTieredPosthoc(unittest.TestCase):
    def fixture(self):
        frozen={"cases":[{"bug":"Toy-1","rows":[
            {"method":"l1_only","budget_proxy":50,"used_proxy_tokens":40,
             "candidate_id_tier":{"A.other()":"L1"}},
            {"method":"tiered_l0","budget_proxy":50,"used_proxy_tokens":20,
             "candidate_id_tier":{"A.bug()":"L0","A.other()":"L0"}},
            {"method":"tiered_upgrade","budget_proxy":50,"used_proxy_tokens":30,
             "candidate_id_tier":{"A.bug()":"L0","A.other()":"L1"}},
        ]}]}
        labels={"scoring_only":True,"selection_frozen_before_labels":True,
                "bugs":{"Toy-1":{"changed_methods":["A.bug()"]}}}
        return frozen,labels

    def test_identifies_signature_gain_but_not_source_gain(self):
        frozen,labels=self.fixture()
        result=score(frozen,labels)
        upgrade=next(r for r in result["rows"] if r["method"]=="tiered_upgrade")
        self.assertEqual(upgrade["identity_recall"],1.0)
        self.assertEqual(upgrade["detailed_source_recall"],0.0)
        self.assertEqual(result["identity_gain_conditions"],2)
        self.assertEqual(result["detail_gain_conditions"],0)

    def test_scoring_never_changes_frozen_selection(self):
        frozen,labels=self.fixture()
        original=dict(frozen["cases"][0]["rows"][1]["candidate_id_tier"])
        score(frozen,labels)
        self.assertEqual(frozen["cases"][0]["rows"][1]["candidate_id_tier"],original)

    def test_labels_must_be_scoring_only(self):
        frozen,labels=self.fixture()
        labels["scoring_only"]=False
        with self.assertRaises(ValueError):
            score(frozen,labels)


if __name__=="__main__":
    unittest.main()
