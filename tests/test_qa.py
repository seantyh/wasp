import wasp.qa as qa

def test_decode_target():
    ret = qa.decode_target_position("這是一個<測>試")
    assert ret[0] == "這是一個測試"
    assert ret[1] == 4
    ret = qa.decode_target_position("這是<另>一個<測>試")
    assert ret[0] == "這是另一個測試"
    assert ret[1] == 2

def test_segment_words():
    q, aes = qa.segment_words("這是一個測試", ["選項沒有", "空的選項", "為什麼你在", "這裡沒有人"])
    assert isinstance(q, list)
    assert isinstance(aes, list)
    assert isinstance(aes[0], list)
    assert len(aes) == 4

def test_max_similarity():
    q_mat, opt_list = qa.max_similarity(["政府", "總統"], [["醫院", "護士"], ["豬", "牛"], ["政黨", "政治"]])
    return q_mat, opt_list