// Included inside cli_self_check after the check lambda; card-free only.
    check({"rad50-tester","--test","combine-ab","--device","0","--iterations","10",
           "--mode","direct","--rows","1","--lease-confirmed"},
          true,"FP16 direct combine one row");
    check({"rad50-tester","--test","combine-ab","--device","1","--iterations","30",
           "--mode","staged","--rows","768","--lease-confirmed"},
          true,"FP16 staged combine max rows");
    check({"rad50-tester","--test","combine-ab","--device","0","--iterations","10",
           "--mode","direct","--lease-confirmed"},
          true,"FP16 combine full row sweep");
    check({"rad50-tester","--test","combine-ab","--device","0","--iterations","10",
           "--mode","direct","--rows","2","--lease-confirmed"},
          false,"FP16 combine rejects unsupported row count");
    check({"rad50-tester","--test","combine-ab","--device","0","--iterations","10",
           "--mode","direct","--bytes","4096","--lease-confirmed"},
          false,"FP16 combine rejects scalar selector");
    check({"rad50-tester","--test","dmabuf-ab","--device","0","--iterations","10",
           "--mode","direct","--rows","16","--lease-confirmed"},
          false,"scalar mode rejects combine rows");
    check({"rad50-tester","--test","combine-ab","--device","0","--iterations","10",
           "--mode","direct","--rows","16"},
          false,"FP16 combine requires lease");
    check({"rad50-tester","--test","combine-ab","--device","0","--iterations","10",
           "--mode","direct","--rows","16","--rows","64","--lease-confirmed"},
          false,"FP16 combine duplicate row selector");
