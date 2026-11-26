from bigcherry.cli import ai_peer_cmds as p

def test_explicit_peer_coverage():
    bdfs=["0000:01:00.0","0000:02:00.0"]
    arches=["gfx1100","gfx1201"]
    header="MANAGED_PAIR "+" ".join(bdfs+arches)
    cases=[f"MANAGED_CASE {m} {s} {1-s} {b} {b//8} pass"
           for b in p.SIZES for s in (0,1) for m in p.METHODS]
    def check(rows,rc=0):
        return p.parse_coverage("\n".join([header,"MANAGED_ACCESS 0 1 1","MANAGED_ACCESS 1 0 1",*rows]),bdfs,arches,rc)
    assert check(cases)["status"]=="pass"
    assert len(check(cases)["cases"])==48
    assert check(cases[:-1])["status"]=="inconclusive"
    assert check([*cases,cases[-1]])["status"]=="inconclusive"
    assert check([cases[0].replace("pass","mismatch"),*cases[1:]],1)["status"]=="mismatch"
    assert check([],2)["status"]=="unavailable"
