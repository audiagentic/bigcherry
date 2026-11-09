    // S1 CPU-only, fixed production-combine comparison. No GPU permissions.
    check({"rad50-tester","--test","combine-cpu","--rows","1",
           "--iterations","10","--helpers","1","--mode","hot"},
          true,"CPU-only one-row hot input");
    check({"rad50-tester","--test","combine-cpu","--rows","768",
           "--iterations","100","--helpers","4","--mode","unique"},
          true,"CPU-only full unique input with four helpers");
    check({"rad50-tester","--test","combine-cpu","--rows","768",
           "--iterations","10","--helpers","2","--mode","hot",
           "--lease-confirmed"},false,"CPU combine never takes a GPU lease");
    check({"rad50-tester","--test","combine-cpu","--rows","768",
           "--iterations","10","--helpers","2","--mode","hot",
           "--device","0"},false,"CPU combine has no device");
    check({"rad50-tester","--test","combine-cpu","--rows","768",
           "--iterations","10","--mode","hot"},
          false,"CPU combine requires explicit helper count");
    check({"rad50-tester","--test","combine-cpu","--rows","768",
           "--iterations","10","--helpers","3","--mode","hot"},
          false,"CPU combine rejects unsupported helper count");
    check({"rad50-tester","--test","combine-cpu","--rows","768",
           "--iterations","10","--helpers","4","--mode","direct"},
          false,"CPU combine demands explicit locality layout");
    check({"rad50-tester","--test","combine-cpu","--rows","7",
           "--iterations","10","--helpers","4","--mode","unique"},
          false,"CPU combine rejects unsupported row shape");
    check({"rad50-tester","--test","combine-cpu","--rows","768",
           "--iterations","9","--helpers","4","--mode","unique"},
          false,"CPU combine rejects too few iterations");
    check({"rad50-tester","--test","combine-cpu","--rows","768",
           "--iterations","10","--helpers","4","--mode","unique",
           "--helpers","1"},false,"CPU combine rejects duplicate helpers");
    check({"rad50-tester","--test","dmabuf-live-parallel","--device","0",
           "--rows","768","--iterations","10","--helpers","2",
           "--lease-confirmed"},false,"GPU mode must reject CPU helpers");
    check({"rad50-tester","--test","cpu-handoff","--mode","hot",
           "--samples","3000","--lease-confirmed"},false,
          "CPU handoff rejects combine input modes");
