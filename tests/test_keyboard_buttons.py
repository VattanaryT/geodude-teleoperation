from geodude_teleoperation.keyboard import dispatch_key


def test_enable_enter():
    assert dispatch_key("\n", step=0.01)[0] == "enable"


def test_t_v_buttons():
    assert dispatch_key("t", step=0.01) == ("button", 0)
    assert dispatch_key("v", step=0.01) == ("button", 1)


def test_twist_w():
    kind, twist = dispatch_key("w", step=0.01)
    assert kind == "twist"
