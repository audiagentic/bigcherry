# Scratch probe: verify project shadow disables bundled rule.
value = float("3.14")
other = int("42")


def load_it(name):
    with open(name) as handle:
        return float(handle.read())
