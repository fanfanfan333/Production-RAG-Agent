from scripts.make_graph_challenge import make_challenge


def test_challenge_is_fixed_grounded_and_does_not_encode_answers_in_names():
    gold = make_challenge()
    assert gold == make_challenge()
    assert len(gold["documents"]) == 40
    multi = [case for case in gold["cases"] if case["category"] == "multihop"]
    assert len(multi) == 20
    assert len({case["answer"] for case in multi}) == 20
    for case in gold["cases"]:
        evidence = [gold["documents"][name] for name in case["expect_sources"]]
        assert case["answer"] in evidence[-1]
        assert case["answer"] not in case["query"]
        assert case["human_review"] == "pending_business_review"
        if case["category"] == "multihop":
            assert len(evidence) == 2
            assert case["answer"] not in evidence[0]
            intermediary = evidence[0].split("投资")[-1].rstrip("。")
            assert intermediary + "位于" in evidence[1]
            assert intermediary not in case["query"]
